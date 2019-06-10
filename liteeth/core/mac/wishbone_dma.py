from math import log2

from migen.fhdl.module import Module
from migen.fhdl.structure import *
from migen.genlib.fsm import FSM, NextState, NextValue

from litex.soc.interconnect import wishbone, stream
from litex.soc.interconnect.csr import CSRStorage, CSRStatus, CSRConstant, AutoCSR
from litex.soc.interconnect.csr_eventmanager import EventManager, EventSourceLevel

from liteeth.core.mac import eth_phy_description


def _get_stream_data_width(stream_desc):
    for field in stream_desc.payload_layout:
        if field[0] == "data":
            if not isinstance(field[1], int):
                raise ValueError("data field in payload layout does not have an integer width")
            return field[1]
    raise ValueError("missing data field in payload layout")


class _WishboneStreamDMABase(object):
    def __init__(self, stream_desc, wb_data_width, wb_adr_width, process_fsm_state):
        # We support only 32-bit or 64-bit wishbone data width, and only 30-bit address
        # width.
        assert wb_data_width in (32, 64)
        assert wb_adr_width == 30

        # Create the Wishbone interface used to access the memory.
        wb_master = wishbone.Interface(wb_data_width, wb_adr_width)
        self.wb_master = wb_master

        # Figure out the data width of the stream from the StreamDescriptor. We require
        # that there is a "data" field in the payload structure.
        self._stream_data_width = _get_stream_data_width(stream_desc)

        # There is no support for data width mismatch between WB and stream for now.
        assert self._stream_data_width == wb_data_width

        # Wishbone data width in bytes and corresponding number of address bits in byte
        # addresses (number of bits to strip to get from byte to WB address).
        wb_data_width_bytes = wb_data_width // 8
        wb_data_width_addr_bits = int(log2(wb_data_width_bytes))

        # Number of bits for representing buffer counts and indices (16 bits allows at
        # most 65535 buffers).
        buffer_index_bits = 16

        # Number of bits for representing buffer sizes in bytes.
        buffer_size_bits = 15

        # Define a CSR constant for the required alignment of the ring buffer and data
        # buffers (in bytes), so the CPU can be sure to align correctly.
        self._csr_mem_align = CSRConstant(wb_data_width_bytes, name="mem_align")

        # The CPU configures the address and size of the ring buffer containing buffer
        # descriptors (see below) by writing values into these registers. ring_addr
        # is the byte address and ring_size_m1 is the number of buffer descriptors minus
        # one. This must not be changed after the CPU has pushed any buffer to the DMA.
        self._csr_ring_addr = CSRStorage(32, name="ring_addr")
        self._csr_ring_size_m1 = CSRStorage(buffer_index_bits, name="ring_size_m1")

        # Each element of the ring buffer ("buffer des") has two 32-bit words:
        # First word:
        #   bits 0-31: Byte address of the buffer. It must be aligned to _csr_mem_align
        #     bytes.
        # Second word:
        #   bits 0-14: Buffer size in bytes. Must be a multiple of mem_align.
        #   bits 15-29: Data size in bytes. For TX all data sizes except the last buffer
        #     in a packet (see bit 31) must be multiples of mem_align. For RX the DMA
        #     guarantees the same.
        #   bit 30: End-of-packet bit. 0 means the packet continues in the next buffer,
        #     1 means this is the last buffer of the packet.
        #   bit 31: Receive error bit. If there is a 1 in any buffer descriptor for a
        #     received packet then the packet is incomplete and should not processed.
        #     Unused for transmission.
        #
        # Data size and end-of-packet bit are updated:
        # - For write DMA, by the CPU before it submits the buffer to the DMA.
        # - For read DMA, by the DMA before it releases the buffer to the CPU.
        #
        # If the wishbone data width is 64-bit then "first word" are the low-order bits
        # and "second word" the high-order bits (i.e. little endian interpretation).

        # Buffer descriptor size in Wishbone words.
        desc_size_words = 64 // wb_data_width

        # General status register.
        #   bit 0: Error state. Do soft reset to restart (see _csr_ctrl).
        #   bit 1: Soft reset in progress (see _csr_ctrl).
        #   bit 2: Interrupt enabled. Default is 0, enable/disable is via _csr_ctrl. If
        #     enabled, the interrupt is generated and latched whenever the DMA releases a
        #     buffer that has the end-of-packet bit set or when _ring_count changes from
        #     non-zero to 0.
        #   bit 3: Interrupt active. Clear via _csr_ctrl. Disabling the interrupt also
        #     clears it.
        self._csr_stat = CSRStatus(8, name='stat')

        # General control register.
        #   bit 0: Write 1 to refresh _csr_ring_count (see the comment there).
        #   bit 1: Write 1 to perform soft reset. This is needed at initialization to
        #     stop operation and reset _ring_count to zero. After requesting soft reset,
        #     wait until the soft reset bit in _csr_stat is cleared.
        #   bit 2: Write 1 to enable the interrupt.
        #   bit 3: Write 1 to disable and clear the interrupt. Disable takes precedence
        #     over enable, but there is no reason to write with both bits set.
        #   bit 4: Write 1 to clear the interrupt.
        self._csr_ctrl = CSRStorage(8, name="ctrl")

        # This register is used to determine the current number of buffers that the DMA
        # owns (buffers to be transmitted or filled with received data). It is updated
        # only when the CPU writes 1 to bit 0 in _csr_ctrl.
        self._csr_ring_count = CSRStatus(buffer_index_bits, name="ring_count")

        # The CPU writes to this register to submit a number of buffers (the written
        # number) to the DMA. Due to its width at most 255 buffers can be submitted with
        # one register write. It must not be wider than 8 bits because then CSR writes
        # are not atomic.
        self._csr_ring_submit = CSRStorage(8, name="ring_submit")

        # Signal definitions and logic start here.

        # General status register logic and component signals.
        self._stat_err = Signal(1)
        self._stat_soft_reset = Signal(1)
        self._interrupt_enabled = Signal(1)
        self._interrupt_active = Signal(1)
        self.comb += [
            self._csr_stat.status[0].eq(self._stat_err),
            self._csr_stat.status[1].eq(self._stat_soft_reset,
            self._csr_stat.status[2].eq(self._interrupt_enabled),
            self._csr_stat.status[3].eq(self._interrupt_active),
        ]

        # Signal which is comb-assigned to indicate when a soft reset is actually being
        # done. It is used to reset _ring_count and also allows the derived class to reset
        # its own states at that time.
        self._doing_soft_reset = Signal(1)

        # The _ring_pos is the index of the first buffer that the DMA owns.
        self._ring_pos = Signal(buffer_index_bits)

        # _ring_count is the current number of buffers that the DMA owns.
        # This number is managed as follows:
        # - When buffers is submitted to the DMA (_csr_ring_submit is written), _ring_count
        #   is incremented by the numbers of submitted buffers.
        # - When a previously submitted buffer is done being processed by the DMA (i.e.
        #   was transmitted or filled with data), _ring_count is decremented by 1.
        self._ring_count = Signal(buffer_index_bits)

        # Logic of for updating _ring_count. _ring_count_inc is comb-assigned in other
        # logic, while _ring_count_dec is sync-assigned to 1 in other logic and is cleared
        # automatically in the next cycle.
        self._next_ring_count = Signal(buffer_index_bits)
        self._ring_count_inc = Signal(8)
        self._ring_count_dec = Signal(1)
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
            _ring_count_dec.eq(0)
        ]

        # Logic for generating the interrupt when _ring_count becomes 0.
        self.sync += [
            If(self._interrupt_enabled and self._ring_count != 0 and self._next_ring_count == 0,
                self._interrupt_active.eq(1)
            )
        ]

        # Partial logic for generating the interrupt when a buffer with the end-of-packet
        # bit set is released.
        self._releasing_last_buffer_in_packet = Signal(1)
        self.sync += [
            If(self._interrupt_enabled and self._releasing_last_buffer_in_packet,
                self._interrupt_active.eq(1)
            )
        ]

        # Logic for updating _csr_ring_count.
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
        # csr_ring_submit then increment ring_count by the written value.
        self.comb += [
            If(self._csr_ring_submit.re,
                self._ring_count_inc.eq(self._csr_ring_submit.storage)
            )
        ]

        # Logic for enabling and disabling the interrupt.
        self.sync += [
            If(self._csr_ctrl.re and self._csr_ctrl.storage[3],
                self._interrupt_enabled.eq(0)
            )
            .Elif(self._csr_ctrl.re and self._csr_ctrl.storage[2],
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
        #   until it's done.
        # - If necessary, update the buffer descriptor (WRITE_DESC).
        # - Release the buffer (already done by derived class if not updating
        #   the buffer descriptor).

        fsm = FSM(reset_state="WAIT_BUFFER")
        self.submodules += fsm

        # Default Wishbone outputs.
        self.comb += [
            wb_master.adr.eq(0),
            wb_master.dat_w.eq(0),
            wb_master.sel.eq(0),
            wb_master.cyc.eq(0),
            wb_master.stb.eq(0),
            wb_master.we.eq(0),
            wb_master.cti.eq(0),
            wb_master.bte.eq(0),
        ]

        # Temporary storage for the descriptor address.
        desc_addr = Signal(wb_adr_width)
        if wb_data_width == 32:
            desc_addr2 = Signal(wb_adr_width)

        # The following are set here based on the descriptor. They can be read from the
        # processing code in the deriver class and also updated during processing.
        # - Buffer address for Wishbone (not byte address).
        # - Buffer size in bytes.
        # - Data size in bytes.
        # - End-of-packet bit.
        self._buffer_addr = Signal(wb_adr_width)
        self._buffer_size = Signal(buffer_size_bits)
        self._buffer_data_size = Signal(buffer_size_bits)
        self._buffer_last = Signal(1)

        # Comb-defined signal for calculating the descriptor address to avoid
        # duplicatication when we need the +1 address.
        desc_addr_calc_sig = Signal(wb_adr_width)
        self.comb += desc_addr_calc_sig.eq(
            (self._csr_ring_addr.storage >> wb_data_width_addr_bits) +
            self._ring_pos * desc_size_words
        )

        fsm.act("WAIT_BUFFER",
            # If soft reset is needed, do it.
            If(self._stat_soft_reset,
                # Set _doing_soft_reset to reset _ring_count and also let the derived
                # class reset its own states.
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
                NextValue(desc_addr, desc_addr_calc_sig),
                If(wb_data_width == 32,
                    NextValue(desc_addr2, desc_addr_calc_sig + 1)
                ),
                # Increment _ring_pos by one (with wrap-around).
                If(self._ring_pos == self._csr_ring_size_m1.storage,
                    NextValue(self._ring_pos, 0)
                ).Else(
                    NextValue(self._ring_pos, self._ring_pos + 1)
                ),
                NextState("READ_DESC_1")
            )
        )

        def save_desc_first_word(desc_first_word):
            return [
                NextValue(self._buffer_addr, desc_first_word >> wb_data_width_addr_bits),
            ]

        def save_desc_second_word(desc_second_word):
            return [
                NextValue(self._buffer_size, desc_second_word & 0x7FFF),
                NextValue(self._buffer_data_size, (desc_second_word >> 15) & 0x7FFF),
                NextValue(self._buffer_last, (desc_second_word >> 30) & 1),
            ]

        fsm.act("READ_DESC_1",
            # Read the descriptor (first word or complete).
            wb_master.adr.eq(desc_addr),
            wb_master.sel.eq((1 << wb_data_width_bytes) - 1)
            wb_master.cyc.eq(1),
            wb_master.stb.eq(1),
            wb_master.we.eq(0),
            If(wb_master.err,
                NextState("ERROR")
            )
            .Elif(wb_master.ack,
                If(wb_data_width == 32,
                    # Store the first word, read the second word now.
                    *save_desc_first_word(wb_master.dat_r),
                    NextState("READ_DESC_2")
                ).Else(
                    # Got the entire descriptor, store it and continug processing.
                    *save_desc_first_word(wb_master.dat_r & 0xFFFFFFFF),
                    *save_desc_second_word(wb_master.dat_r >> 32),
                    NextState(process_fsm_state)
                )
            )
        )

        if wb_data_width == 32:
            # Read the second word of the descriptor.
            fsm.act("READ_DESC_2",
                wb_master.adr.eq(desc_addr2),
                wb_master.sel.eq((1 << wb_data_width_bytes) - 1),
                wb_master.cyc.eq(1),
                wb_master.stb.eq(1),
                wb_master.we.eq(0),
                If(wb_master.err,
                    NextState("ERROR")
                )
                .Elif(wb_master.ack,
                    # Store the second word and continug processing.
                    *save_desc_second_word(wb_master.dat_r),
                    NextState(process_fsm_state)
                )
            )
        
        # By going to process_fsm_state we pass control to the derived class.
        # The derived class will eventually transition to one of:
        # - WAIT_BUFFER: If it's done and the descriptor should not be updated. In this
        #   case the derived class must decrement _ring_count by one, by sync-assigning
        #   _ring_count_dec to 1 exactly once. If the buffer has the end-of-packet bit set
        #   then _releasing_last_buffer_in_packet must also be sync-assigned to 1 in the
        #   same cycle. This must be done only after the buffer is no longer being used!
        # - WRITE_DESC: If it's done and the descriptor should be updated. In this case
        #   this class will take care of decrementing _ring_count by one and setting
        #   _releasing_last_buffer_in_packet if needed, not the derived class.
        # - ERROR: If there was an error. It does not matter whether _ring_count was
        #   decremented.
        # - When _stat_soft_reset is active, transition directly to WAIT_BUFFER is
        #   allowed, where the soft reset will be done.

        # By the time the derived class returns control to 

        # If the transition was to WRITE_DESC, the base class assigned the value
        # to write to the second word of the descriptor this signal.
        self._update_descriptor_value = Signal(32)

        fsm.act("WRITE_DESC",
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
                NextValue(self._ring_count_dec, 1),
                # Generate the interrupt for the last buffer in a packet (if enabled).
                If(self._update_descriptor_value[30], # end-of-packet bit
                    self._releasing_last_buffer_in_packet.eq(1)
                ),
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

        fsm = self.fsm
        wb_master = self.wb_master

        wb_data_width_bytes = wb_data_width // 8

        # This signal is used to communicate an error in the pipeline to the driving
        # FSM state DMA_READ_PIPELINE.
        wb_error = Signal(1)

        # Pipeline stage 1: check the remaining number of bytes and calculate a bunch
        # of stuff needed in further stages.
        p1_ready = Signal(1)
        p1_valid = Signal(1)
        p1_buffer_addr = Signal(wb_adr_width)
        p1_data_sel = Signal(wb_data_width_bytes)
        p1_last_be = Signal(wb_data_width_bytes)
        p1_last_word_in_packet = Signal(1)

        # Clear p1_valid when data is tranferred if there is no next data. When there
        # is next data this assignment is overridden in code below.
        self.sync += If(p1_valid and p1_ready, p1_valid.eq(0))

        fsm.act("DMA_READ_PIPELINE",
            If(wb_error,
                # Wishbone read error, handle by going to ERROR state. There is no need
                # sync with the pipeline because the reading is done right in the next
                # pipeline stage and after a read error the data is accepted (i.e.
                # p1_ready must also be true here).
                NextState("ERROR")
            )
            .Elif(self._buffer_data_size == 0,
                # Before we can continue we need to make sure that the next pipeline
                # stage is done with reading from this buffer. But it is simple, just
                # check p1_valid.
                If(not p1_valid,
                    # We're done, make sure the _ring_count is decremented and continue
                    # with the next buffers.
                    NextValue(self._ring_count_dec, 1),
                    # Generate the interrupt for the last buffer in a packet (if enabled).
                    If(self._buffer_last, # end-of-packet bit
                        self._releasing_last_buffer_in_packet.eq(1)
                    ),
                    NextState("WAIT_BUFFER")
                )
            ).
            Else(
                # Generate data for the next pipeline stage.
                If(not p1_valid or p1_ready,
                    NextValue(p1_valid, 1),
                    NextValue(p1_buffer_addr, self._buffer_addr),
                    NextValue(self._buffer_addr, self._buffer_addr + 1),
                    If(self._buffer_data_size <= wb_data_width_bytes,
                        NextValue(self._buffer_data_size, 0),
                        NextValue(p1_data_sel, (1 << self._buffer_data_size) - 1),
                        NextValue(p1_last_be, 1 << (self._buffer_data_size - 1)),
                        NextValue(p1_last_word_in_packet, self._buffer_last)
                    ).Else(
                        NextValue(self._buffer_data_size, self._buffer_data_size - wb_data_width_bytes),
                        NextValue(p1_data_sel, (1 << wb_data_width_bytes) - 1),
                        NextValue(p1_last_be, 0),
                        NextValue(p1_last_word_in_packet, 0)
                    )
                )
            )
        )

        # Pipeline stage 2: Read the data word from Wishbone.
        p2_ready = Signal(1)
        p2_valid = Signal(1)
        p2_last_be = Signal(wb_data_width_bytes)
        p2_last_word_in_packet = Signal(1)
        p2_data = Signal(wb_data_width)

        self.sync += If(p2_valid and p2_ready, p2_valid.eq(0))

        self.comb += [
            If(p1_valid and (not p2_valid or p2_ready),
                wb_master.adr.eq(p1_buffer_addr),
                wb_master.sel.eq(p1_data_sel),
                wb_master.cyc.eq(1),
                wb_master.stb.eq(1),
                wb_master.we.eq(0),
                If(wb_master.err,
                    wb_error.eq(1)
                ),
                If(wb_master.err or wb_master.ack,
                    p1_ready.eq(1)
                )
            )
        ]
        self.sync += [
            If(p1_valid and (not p2_valid or p2_ready),
                If(wb_master.ack,
                    p2_valid.eq(1),
                    p2_last_be.eq(p1_last_be),
                    p2_last_word_in_packet.eq(p1_last_word_in_packet),
                    p2_data.eq(wb_master.dat_r)
                )
            )
        ]

        # Finally connect the output of the last stage with the stream. The only
        # nontrivial thing is generation of source.first.

        # This signal indicates whether the next data word will start a new packet. It
        # is a state that persists across processing of individual buffers.
        first_word_in_packet = Signal(1, reset=1)

        # Logic for first_word_in_packet.
        self.sync += [
            # Reset to 1 at soft reset.
            If(self._doing_soft_reset,
                first_word_in_packet.eq(1)
            )
            # Update whenever to source transfer occurs.
            .Elif(p2_valid and p2_ready,
                first_word_in_packet.eq(p2_last_word_in_packet)
            )
        ]

        self.comb += [
            p2_ready.eq(source.ready),
            source.valid.eq(p2_valid),
            source.first.eq(first_word_in_packet),
            source.last.eq(p2_last_word_in_packet),
            source.payload.data.eq(p2_data),
            source.payload.last_be.eq(p2_last_be),
            source.payload.error.eq(0)
        ]


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

