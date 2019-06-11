from math import log2

from migen.fhdl.module import Module
from migen.fhdl.structure import *
from migen.genlib.fsm import FSM, NextState, NextValue

from litex.soc.interconnect import wishbone, stream
from litex.soc.interconnect.csr import CSRStorage, CSRStatus, CSRConstant, AutoCSR
from litex.soc.interconnect.csr_eventmanager import EventManager, EventSourceLevel

from liteeth.core.mac import eth_phy_description


def _check_stream_and_get_data_width(stream_desc):
    payload_fields = dict((field[0], field) for field in stream_desc.payload_layout)

    if "data" not in payload_fields:
        raise ValueError("missing data field in payload layout")
    if not isinstance(payload_fields["data"][1], int):
        raise ValueError("data field in payload layout does not have integer width")
    
    if "last_be" not in payload_fields:
        raise ValueError("missing last_be field in payload layout")

    if "error" not in payload_fields:
        raise ValueError("missing error field in payload layout")

    return payload_fields["data"][1]


def _auto_clear_valid(module, ready, valid):
    # Clear valid when data is tranferred if there is no next data. When there
    # is next data this assignment is overridden in subsequent code.
    module.sync += [
        If(valid and ready,
            valid.eq(0)
        )
    ]


class _WishboneStreamDMABase(object):
    def __init__(self, stream_desc, wb_data_width, wb_adr_width, process_fsm_state):
        # We support only 32-bit or 64-bit wishbone data width.
        assert wb_data_width in (32, 64)

        # Create the Wishbone interface used to access the memory.
        wb_master = wishbone.Interface(wb_data_width, wb_adr_width)
        self.wb_master = wb_master

        # Figure out the data width of the stream from the StreamDescriptor.
        self._stream_data_width = _check_stream_and_get_data_width(stream_desc)

        # There is no support for data width mismatch between WB and stream for now.
        assert self._stream_data_width == wb_data_width

        # Wishbone data width in bytes and corresponding number of address bits in byte
        # addresses (number of bits to strip to get from byte to WB address).
        wb_data_width_bytes = wb_data_width // 8
        self._wb_data_width_bytes = wb_data_width_bytes
        wb_data_width_addr_bits = int(log2(wb_data_width_bytes))

        # We support only WB address width equivalent to 32-bit byte address.
        assert wb_adr_width == 32 - wb_data_width_addr_bits

        # Number of bits for representing buffer counts and indices (16 bits allows at
        # most 65535 buffers). This can be changed without much concern.
        buffer_index_bits = 16

        # Number of bits for representing buffer sizes in bytes (14 bits allows jumbo
        # frames with MTU 9000). This must not be more than 15 because then we run out of
        # bits in the buffer descriptor!
        buffer_size_bits = 14
        assert buffer_size_bits <= 15

        # CSR constant for the required alignment (in bytes) of the descriptors ring buffer
        # and data buffers, so the CPU can be sure to align correctly.
        self._csr_mem_align = CSRConstant(wb_data_width_bytes, name="mem_align")

        # The CPU configures the address and size of the ring buffer containing buffer
        # descriptors (see below) by writing values into these registers. _csr_ring_addr
        # is the byte address and _csr_ring_size_m1 is the number of buffer descriptors
        # minus one. This must not be changed while the DMA is responsible for any buffers
        # (_ring_count > 0).
        self._csr_ring_addr = CSRStorage(32, name="ring_addr")
        self._csr_ring_size_m1 = CSRStorage(buffer_index_bits, name="ring_size_m1")

        # Each buffer descriptor in the ring buffer is made of two 32-bit words:
        # First word:
        #   bits 0..31: Byte address of the buffer. It must be aligned to _csr_mem_align
        #     bytes.
        # Second word:
        #   bits 0..14: Buffer size (in bytes). It must be set by the CPU and mut be a
        #     multiple of mem_align. Currently only used for RX but should also be set
        #     meaningfully for TX.
        #   bits 15..29: Data size (in bytes). For TX this must be set by the CPU, and all
        #     data sizes except for the last buffer in a packet (see bit 31) must be
        #     multiples of _csr_mem_align. For RX the DMA sets this before the buffer is
        #     returned to the CPU, and for all but the last buffer in a packet it will be a
        #     multiple of _csr_mem_align.
        #   bit 30: End-of-packet bit. 0 means the packet continues in the next buffer,
        #     1 means this is the last buffer in the packet. For TX it must be set by the
        #     CPU, for RX the DMA sets this before the buffer is returned to the CPU.
        #   bit 31: Receive error bit. For RX, if this is 1 in any buffer descriptor of a
        #     received packet then the packet is incomplete and should not processed.
        #     Unused for TX.
        #
        # If the wishbone data width is 64-bit then "first word" are the low-order bits
        # and "second word" the high-order bits (i.e. little endian interpretation).
        #
        # The CPU must correctly set the entire descriptor before submitting the buffer
        # to the DMA. For RX, the DMA also updates the descriptor before returning the
        # buffer to the CPU. In any case, the DMA does not read any descriptor before the
        # buffer is submitted.

        # Buffer descriptor size in Wishbone words (1 or 2).
        desc_size_words = 64 // wb_data_width

        # General status register.
        #   bit 0: Error state, DMA is stopped. Do soft reset to restart (see _csr_ctrl).
        #   bit 1: Soft reset in progress (see _csr_ctrl).
        #   bit 2: Interrupt enabled. Default is 0, enable/disable is via _csr_ctrl. If
        #     enabled, the interrupt is generated and latched whenever the DMA releases a
        #     buffer that has the end-of-packet bit set (i.e. packet transmitted/received)
        #     or when _ring_count changes from non-zero to 0.
        #   bit 3: Interrupt active. Clear via _csr_ctrl. Disabling the interrupt also
        #     clears it.
        self._csr_stat = CSRStatus(8, name='stat')

        # General control register.
        #   bit 0: Write 1 to refresh _csr_ring_count (see the comment there).
        #   bit 1: Write 1 to perform a soft reset. This is needed at initialization to
        #     stop operation and reset the buffer states. After requesting soft reset,
        #     wait until the soft reset bit in _csr_stat is cleared.
        #   bit 2: Write 1 to enable the interrupt.
        #   bit 3: Write 1 to disable and clear the interrupt. Disable takes precedence
        #     over enable, but there is no reason to write with both bits set.
        #   bit 4: Write 1 to clear the interrupt.
        self._csr_ctrl = CSRStorage(8, name="ctrl")

        # This register is used to determine the current number of buffers that the DMA
        # owns (buffers to be transmitted or filled with received data). It is updated
        # to the current value of _ring_count only when the CPU requests the update via
        # _csr_ctrl (to allow atomic read).
        self._csr_ring_count = CSRStatus(buffer_index_bits, name="ring_count")

        # The CPU writes to this register to submit a number of buffers (the written
        # number) to the DMA. Due to its width at most 255 buffers can be submitted with
        # one register write. It must not be wider than 8 bits because then CSR writes
        # are not atomic. Note that for TX, all buffers making up a packet must be
        # submitted as part of one write to this register, else there is a risk of FIFO
        # underrun.
        self._csr_ring_submit = CSRStorage(8, name="ring_submit")

        # Signal definitions and logic start here.

        # General status register assignment from component signals.
        self._stat_err = Signal(1) # comb-assigned from ERROR state
        self._stat_soft_reset = Signal(1)
        self._interrupt_enabled = Signal(1)
        self._interrupt_active = Signal(1)
        self.comb += [
            self._csr_stat.status[0].eq(self._stat_err),
            self._csr_stat.status[1].eq(self._stat_soft_reset),
            self._csr_stat.status[2].eq(self._interrupt_enabled),
            self._csr_stat.status[3].eq(self._interrupt_active),
        ]

        # Signal which is comb-assigned to indicate when a soft reset is actually being
        # done. It is used to reset _ring_count and also allows the derived class to reset
        # its own states at that time.
        self._doing_soft_reset = Signal(1)

        # The _ring_pos is the index of the first buffer that the DMA owns. It is reset to
        # 0 at soft reset and is incremented by one each time the DMA starts processing a
        # buffer, wrapping around to 0 after _csr_ring_size_m1.
        self._ring_pos = Signal(buffer_index_bits)

        # _ring_count is the current number of buffers that the DMA owns.
        # This number is managed as follows:
        # - It is reset to 0 at soft reset.
        # - When buffers is submitted to the DMA (_csr_ring_submit is written), _ring_count
        #   is incremented by the numbers of submitted buffers.
        # - When a previously submitted buffer is released to the CPU (after being
        #   transmiitted or filled with received data), _ring_count is decremented by 1.
        self._ring_count = Signal(buffer_index_bits)

        # Logic for updating _ring_count. _ring_count_inc is comb-assigned in other logic,
        # while _ring_count_dec is sync-assigned to 1 in other logic and is cleared
        # automatically in the next cycle.
        self._ring_count_inc = Signal(8)
        self._ring_count_dec = Signal(1)
        self._next_ring_count = Signal(buffer_index_bits)
        self.comb += [
            If(self._doing_soft_reset,
                self._next_ring_count.eq(0)
            ).Else(
                self._next_ring_count.eq(
                    self._ring_count + self._ring_count_inc - self._ring_count_dec)
            )
        ]
        self.sync += [
            self._ring_count.eq(self._next_ring_count),
            self._ring_count_dec.eq(0)
        ]

        # Logic for generating the interrupt when _ring_count becomes 0.
        self.sync += [
            If(self._interrupt_enabled and self._ring_count != 0 and self._next_ring_count == 0,
                self._interrupt_active.eq(1)
            )
        ]

        # Partial logic for generating the interrupt when a buffer with the end-of-packet
        # bit set is released. _releasing_last_buffer_in_packet is sync-assigned from
        # other code when this happens.
        self._releasing_last_buffer_in_packet = Signal(1)
        self.sync += [
            If(self._interrupt_enabled and self._releasing_last_buffer_in_packet,
                self._interrupt_active.eq(1)
            ),
            self._releasing_last_buffer_in_packet.eq(0) # clear automatically
        ]

        # Logic for updating _csr_ring_count when requested by the CPU.
        self.sync += [
            If(self._csr_ctrl.re and self._csr_ctrl.storage[0],
                self._csr_ring_count.status.eq(self._ring_count)
            )
        ]
        
        # Latch a soft reset request.
        self.sync += [
            If(self._csr_ctrl.re and self._csr_ctrl.storage[1],
                self._stat_soft_reset.eq(1)
            )
        ]

        # Implementation of _csr_ring_submit. If a value is being written to
        # _csr_ring_submit then increment _ring_count by the written value.
        self.comb += [
            If(self._csr_ring_submit.re,
                self._ring_count_inc.eq(self._csr_ring_submit.storage)
            )
        ]

        # Logic for enabling and disabling the interrupt via _csr_ctrl.
        self.sync += [
            If(self._csr_ctrl.re and self._csr_ctrl.storage[3],
                # Disable the interrupt.
                self._interrupt_enabled.eq(0)
            )
            .Elif(self._csr_ctrl.re and self._csr_ctrl.storage[2],
                # Enable the interrupt.
                self._interrupt_enabled.eq(1)
            )
        ]

        # Logic for clearing the interrupt. Note that this takes precedence over generating
        # the interrupt, but it shouldn't matter what the behavior is.
        self.sync += [
            If(self._csr_ctrl.re and (self._csr_ctrl.storage[3] or self._csr_ctrl.storage[4]),
                self._interrupt_active.eq(0)
            )
        ]

        # Interrupt setup.
        self.submodules.ev = EventManager()
        self.ev.interrupt = EventSourceLevel()
        self.ev.finalize()
        self.comb += self.ev.interrupt.trigger.eq(self._interrupt_active)
        
        # FSM which does the following in a loop:
        # - Wait for a buffer to be available (WAIT_BUFFER).
        # - Read the buffer descriptor (READ_DESC_1, READ_DESC_2).
        # - Pass control to the derived class to do the DMA read/write and wait
        #   until it's done (process_fsm_state).
        # - If necessary, update the buffer descriptor (WRITE_DESC).
        # - Release the buffer (already done by derived class if not updating
        #   the buffer descriptor).

        fsm = FSM(reset_state="WAIT_BUFFER")
        self.submodules._fsm = fsm

        # Temporary storage for the descriptor address.
        desc_addr = Signal(wb_adr_width)

        # The following are set based on the descriptor. They can be read from the
        # processing code in the derived class and also updated during processing
        # (updating does not change the behavior of the base class).
        # - Buffer address for Wishbone (not byte address).
        # - Buffer size in bytes.
        # - Data size in bytes.
        # - End-of-packet bit.
        self._buffer_addr = Signal(wb_adr_width)
        self._buffer_size = Signal(buffer_size_bits)
        self._buffer_data_size = Signal(buffer_size_bits)
        self._buffer_last = Signal(1)

        fsm.act("WAIT_BUFFER",
            # If soft reset is needed, do it.
            If(self._stat_soft_reset,
                # Set _doing_soft_reset in order to reset _ring_count and also let the
                # derived class reset its own states.
                self._doing_soft_reset.eq(1),
                # Reset _ring_pos.
                NextValue(self._ring_pos, 0),
                # Clear _stat_soft_reset now that we have done the soft reset.
                NextValue(self._stat_soft_reset, 0)
            )
            # If we have a buffer available, prepare some things and continue.
            # We must take into account the current _ring_count_dec because a decrement
            # may have been requested just prior to entry to the WAIT_BUFFER state.
            .Elif(self._ring_count - self._ring_count_dec > 0,
                # Calculate the descriptor address based on _csr_ring_addr and _ring_pos.
                NextValue(desc_addr,
                    self._csr_ring_addr.storage[wb_data_width_addr_bits:] +
                    self._ring_pos * desc_size_words),
                # Increment _ring_pos by one (with wrap-around).
                If(self._ring_pos == self._csr_ring_size_m1.storage,
                    NextValue(self._ring_pos, 0)
                ).Else(
                    NextValue(self._ring_pos, self._ring_pos + 1)
                ),
                # Continue reading the buffer descriptor.
                NextState("READ_DESC_1")
            )
        )

        def save_desc_first_word(desc_first_word):
            return [
                NextValue(self._buffer_addr, desc_first_word[wb_data_width_addr_bits:]),
            ]

        def save_desc_second_word(desc_second_word):
            return [
                NextValue(self._buffer_size, desc_second_word[0:buffer_size_bits]),
                NextValue(self._buffer_data_size, desc_second_word[15:15+buffer_size_bits]),
                NextValue(self._buffer_last, desc_second_word[30]),
            ]

        fsm.act("READ_DESC_1",
            # Read the descriptor (first word or complete).
            wb_master.adr.eq(desc_addr),
            wb_master.sel.eq((1 << wb_data_width_bytes) - 1),
            wb_master.cyc.eq(1),
            wb_master.stb.eq(1),
            wb_master.we.eq(0),
            If(wb_master.err,
                NextState("ERROR")
            )
            .Elif(wb_master.ack,
                If(wb_data_width == 32,
                    # Store the first word, continue reading the second word.
                    *save_desc_first_word(wb_master.dat_r),
                    NextValue(desc_addr, desc_addr + 1),
                    NextState("READ_DESC_2")
                ).Else(
                    # Store the descriptor and continue processing in the derived class.
                    *save_desc_first_word(wb_master.dat_r[0:32]),
                    *save_desc_second_word(wb_master.dat_r[32:64]),
                    NextState(process_fsm_state)
                )
            )
        )

        if wb_data_width == 32:
            # Read the second word of the descriptor.
            fsm.act("READ_DESC_2",
                wb_master.adr.eq(desc_addr),
                wb_master.sel.eq((1 << wb_data_width_bytes) - 1),
                wb_master.cyc.eq(1),
                wb_master.stb.eq(1),
                wb_master.we.eq(0),
                If(wb_master.err,
                    NextState("ERROR")
                )
                .Elif(wb_master.ack,
                    # Store the second word aand continue processing in the derived class.
                    *save_desc_second_word(wb_master.dat_r),
                    NextState(process_fsm_state)
                )
            )
        
        # By going into process_fsm_state we pass control to the derived class.
        # The derived class will eventually transition to one of:
        # - WAIT_BUFFER: If it's done and the descriptor should not be updated. In this
        #   case the derived class must decrement _ring_count by one, by sync-assigning
        #   1 to _ring_count_dec exactly once. If the buffer has the end-of-packet bit set
        #   then also 1 must be sync-assigned to _releasing_last_buffer_in_packet in the
        #   same cycle. This must be done only after the buffer is no longer being used
        #   by the DMA!
        # - WRITE_DESC: If it's done and the descriptor should be updated. In this case
        #   this class will take care of decrementing _ring_count by one and setting
        #   _releasing_last_buffer_in_packet if needed, not the derived class.
        # - ERROR: If there was an error. It does not matter whether the _ring_count was
        #   decremented.
        # - When _stat_soft_reset is active, transition directly to WAIT_BUFFER is
        #   allowed, where the soft reset will be done.

        # If the transition was to WRITE_DESC, the base class assigns the value to
        # be written to the second word of the descriptor to this signal.
        self._update_descriptor_value = Signal(32)

        fsm.act("WRITE_DESC",
            # If wb_data_width == 32 then desc_addr is the address of the second
            # word of the descriptor (it was incremented by 1 at entry to READ_DESC_2).
            # If wb_data_width == 64 then desc_addr is the original descriptor address.
            wb_master.adr.eq(desc_addr),
            wb_master.dat_w.eq(
                self._update_descriptor_value if wb_data_width == 32
                else (self._update_descriptor_value << 32)),
            wb_master.sel.eq(0b1111 << (0 if wb_data_width == 32 else 4)),
            wb_master.cyc.eq(1),
            wb_master.stb.eq(1),
            wb_master.we.eq(1),
            If(wb_master.err,
                NextState("ERROR")
            )
            .Elif(wb_master.ack,
                # Decrement _ring_count by 1, in the next cycle.
                NextValue(self._ring_count_dec, 1),
                # Generate the interrupt for the last buffer in a packet (if enabled).
                If(self._update_descriptor_value[30], # end-of-packet bit
                    NextValue(self._releasing_last_buffer_in_packet, 1)
                ),
                # Continue processing subsequent buffers.
                NextState("WAIT_BUFFER")
            )
        )

        fsm.act("ERROR",
            # Error bit in _csr_stat is active in this state.
            self._stat_err.eq(1),
            # Stay here waiting for soft reset. Once found, just go to WAIT_BUFFER where
            # it will actually be handled.
            If(self._stat_soft_reset,
                NextState("WAIT_BUFFER")
            )
        )


class WishboneStreamDMARead(Module, AutoCSR, _WishboneStreamDMABase):
    def __init__(self, stream_desc, wb_data_width=32, wb_adr_width=30):
        _WishboneStreamDMABase.__init__(
            self, stream_desc, wb_data_width, wb_adr_width, "DMA_READ_PIPELINE")

        source = stream.Endpoint(stream_desc)
        self.source = source

        wb_master = self.wb_master
        wb_data_width_bytes = self._wb_data_width_bytes
        fsm = self._fsm

        # This signal is used to communicate an error in the pipeline to the driving
        # FSM state DMA_READ_PIPELINE.
        wb_error = Signal(1)

        # Signals for communication with the next pipeline stage.
        p1_ready = Signal(1)
        p1_valid = Signal(1)
        _auto_clear_valid(self, p1_ready, p1_valid)
        p1_buffer_addr = Signal(wb_adr_width)
        p1_data_sel = Signal(wb_data_width_bytes)
        p1_last_be = Signal(wb_data_width_bytes)
        p1_last = Signal(1)

        # Pipeline stage 1: Check the remaining number of bytes and calculate a bunch of
        # stuff needed in subsequent pipeline stages.
        fsm.act("DMA_READ_PIPELINE",
            If(wb_error,
                # Wishbone read error, handle by going to ERROR state. There is no need
                # sync with the pipeline because the data is accepted when an error occurs
                # and there is no additional pipeline stage in between.
                NextState("ERROR")
            )
            .Elif(self._buffer_data_size == 0,
                # Before we can continue we need to wait until the next pipeline stage is
                # done reading from this buffer.
                If(not p1_valid,
                    # Make sure that _ring_count will be decremented.
                    NextValue(self._ring_count_dec, 1),
                    # Generate the interrupt for the last buffer in a packet (if enabled).
                    If(self._buffer_last,
                        NextValue(self._releasing_last_buffer_in_packet, 1)
                    ),
                    # Continue processing subsequent buffers.
                    NextState("WAIT_BUFFER")
                )
            )
            .Else(
                # Wait until the next pipeline stage can accept our data.
                If(not p1_valid or p1_ready,
                    # Pass data for the next stage...
                    NextValue(p1_valid, 1),
                    # Forward the buffer address.
                    NextValue(p1_buffer_addr, self._buffer_addr),
                    # Increment the buffer address.
                    NextValue(self._buffer_addr, self._buffer_addr + 1),
                    # Check if this is the last data in this buffer.
                    If(self._buffer_data_size <= wb_data_width_bytes, # last data
                        # Calculate data_sel and last_be based on the remaining number of
                        # bytes.
                        NextValue(p1_data_sel, (1 << self._buffer_data_size) - 1),
                        NextValue(p1_last_be, 1 << (self._buffer_data_size - 1)),
                        # This is the last data in the packet if this is the last buffer.
                        NextValue(p1_last, self._buffer_last),
                        # Update _buffer_data_size to 0, there is no more data in this buffer.
                        NextValue(self._buffer_data_size, 0)
                    ).Else( # not last data
                        # Set data_sel and last_be for a full data.
                        NextValue(p1_data_sel, (1 << wb_data_width_bytes) - 1),
                        NextValue(p1_last_be, 0),
                        # This is not the last data in the packet.
                        NextValue(p1_last, 0),
                        # Decrement _buffer_data_size by the number of bytes in this data.
                        NextValue(self._buffer_data_size, self._buffer_data_size - wb_data_width_bytes)
                    )
                )
            )
        )

        # Signals for communication with the next pipeline stage.
        p2_ready = Signal(1)
        p2_valid = Signal(1)
        _auto_clear_valid(self, p2_ready, p2_valid)
        p2_last_be = Signal(wb_data_width_bytes)
        p2_last_word_in_packet = Signal(1)
        p2_data = Signal(wb_data_width)

        # Pipeline stage 2: Read the data word from Wishbone.
        wb_read_fsm = FSM()
        self.submodules._wb_read_fsm = wb_read_fsm
        wb_read_fsm.act("DEF",
            If(p1_valid and (not p2_valid or p2_ready),
                wb_master.adr.eq(p1_buffer_addr),
                wb_master.sel.eq(p1_data_sel),
                wb_master.cyc.eq(1),
                wb_master.stb.eq(1),
                wb_master.we.eq(0),
                If(wb_master.err,
                    p1_ready.eq(1),
                    wb_error.eq(1)
                )
                .Elif(wb_master.ack,
                    p1_ready.eq(1),
                    NextValue(p2_valid, 1),
                    NextValue(p2_last_be, p1_last_be),
                    NextValue(p2_last_word_in_packet, p1_last),
                    NextValue(p2_data, wb_master.dat_r)
                )
            )
        )

        # Pipeline stage 3: Transfer the data to the stream (this isn't much of a stage
        # because it mostly just connects signals).
        stream_output_fsm = FSM()
        self.submodules._stream_output_fsm = stream_output_fsm
        first_word_in_packet = Signal(1, reset=1)
        stream_output_fsm.act("DEF",
            # Connect the stream and the previous pipeline stage.
            p2_ready.eq(source.ready),
            source.valid.eq(p2_valid),
            source.last_be.eq(p2_last_be),
            source.last.eq(p2_last_word_in_packet),
            source.data.eq(p2_data),

            # Generate the error signal. TODO
            source.error.eq(0),

            # Generate source.first. TODO
            source.first.eq(first_word_in_packet),
            If(self._doing_soft_reset,
                NextValue(first_word_in_packet, 1)
            )
            .Elif(p2_valid and p2_ready,
                NextValue(first_word_in_packet, p2_last_word_in_packet)
            )
        )


class WishboneStreamDMAWrite(Module, AutoCSR, _WishboneStreamDMABase):
    def __init__(self, stream_desc, wb_data_width=32, wb_adr_width=30):
        _WishboneStreamDMABase.__init__(
            self, stream_desc, wb_data_width, wb_adr_width, "CHECK_BUF_POS")

        self.sink = stream.Endpoint(stream_desc)


class LiteEthMACWishboneDMA(Module, AutoCSR):
    def __init__(self, eth_dw, endianness, wb_data_width=32, wb_adr_width=30):

        stream_desc = eth_phy_description(eth_dw)

        self.submodules.dma_tx = WishboneStreamDMARead(stream_desc, wb_data_width, wb_adr_width)
        self.submodules.dma_rx = WishboneStreamDMAWrite(stream_desc, wb_data_width, wb_adr_width)

        self.wb_master_tx = self.dma_tx.wb_master
        self.wb_master_rx = self.dma_rx.wb_master

        self.sink = self.dma_tx.sink
        self.source = self.dma_rx.source

