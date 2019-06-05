from math import log2

from migen.fhdl.module import Module
from migen.fhdl.structure import *
from migen.genlib.fsm import FSM, NextState, NextValue

from litex.soc.interconnect import wishbone, stream
from litex.soc.interconnect.csr import CSRStorage, CSRStatus, CSRConstant, AutoCSR

from liteeth.core.mac import eth_phy_description


def _get_stream_data_width(stream_desc):
    for field in stream_desc.payload_layout:
        if field[0] == "data":
            if not isinstance(field[1], int):
                raise ValueError("data field in payload layout does not have an integer width")
            return field[1]
    raise ValueError("missing data field in payload layout")


class _WishboneStreamDMABase(object):
    def __init__(self, stream_desc, wb_data_width, wb_adr_width):
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

        # Figure out the memory alignment (based on the widhbone data width) in bytes and
        # the associated number of low address bits.
        mem_align_bytes = wb_data_width // 8
        mem_align_addr_bits = int(log2(mem_align_bytes))

        # Number of bits for representing buffer counts and indices (16 bits allows at
        # most 65535 buffers).
        buffer_index_bits = 16

        # Number of bits for representing buffer sizes in bytes.
        buffer_size_bits = 15

        # Define a CSR constant for the required alignment of the ring buffer and data
        # buffers, so the CPU can be sure to align correctly.
        self._csr_mem_align = CSRConstant(mem_align_bytes, name="mem_align")

        # The CPU configures the address and size of the ring buffer containing buffer
        # descriptors (see below) by writing values into these registers. ring_addr
        # is the byte address and ring_size_m1 is the number of buffer descriptors minus
        # one. This must not be changed after the CPU has pushed any buffer to the DMA.
        self._csr_ring_addr = CSRStorage(32, name="ring_addr")
        self._csr_ring_size_m1 = CSRStorage(buffer_index_bits, name="ring_size_m1")

        # Each element of the ring buffer ("buffer des") has two 32-bit words:
        # First word:
        #  bits 0-31: The byte address of the buffer.
        # Second word:
        #   bits 0-14: Buffer size in bytes. Must be a multiple of mem_align.
        #   bits 15-29: Data size in bytes.
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

        # These registers allow the CPU to determine the current number of buffers that
        # the DMA owns (i.e. buffers to be transmitted or filled with received data).
        # The CPU first writes 1 to ring_count_update and then reads ring_count_value.
        # See _ring_count below for the meaning of this value.
        self._csr_ring_count_update = CSRStorage(1, name="ring_count_update")
        self._csr_ring_count_value = CSRStatus(buffer_index_bits, name="ring_count_value")

        # The CPU writes to this register to submit a number of buffers (the written
        # number) to the DMA. Due to its width at most 255 buffers can be submitted with
        # one register write. It must not be wider than 8 bits because then CSR writes
        # are not atomic.
        self._csr_ring_submit = CSRStorage(8, name="ring_submit")

        # This CSR is used by the CPU to stop DMA operation and reset the buffer position
        # to zero. The CPU writes 1 into it and then waits until ring_count (see above)
        # is zero. The driver must do this at initialization if there is any possiblilty
        # that the DMA is in a non-reset state.
        self._csr_soft_reset = CSRStorage(1, name="soft_reset")

        # Signal definitions and logic start here.

        # The _ring_pos is the index of the first buffer that the DMA owns.
        self._ring_pos = Signal(buffer_index_bits)

        # _ring_count is the current number of buffers that the DMA owns.
        # This number is managed as follows:
        # - When buffers is submitted to the DMA (_csr_ring_submit is written), _ring_count
        #   is incremented by the numbers of submitted buffers.
        # - When a previously submitted buffer is done being processed by the DMA (i.e.
        #   was transmitted or filled with data), _ring_count is decremented by 1.
        self._ring_count = Signal(buffer_index_bits)

        # This signal is uses to latch a soft reset request.
        self._latch_soft_reset = Signal(1)
        self.sync += [
            If(self._csr_soft_reset.re,
                self._latch_soft_reset.eq(1)
            )
        ]

        # This internal signal is set when the soft reset should be done. It has a comb
        # default value 0 and is assigned to 1 in the FSM when the soft reset should
        # be done.
        self._do_soft_reset = Signal(1)
        self.comb += self._do_soft_reset.eq(0)

        # Implementation of _csr_ring_count_update and _csr_ring_count_value. The value
        # of _csr_ring_count_value is updated the the actual _ring_count when 
        # _csr_ring_count_update is written.
        self.sync += [
            If(self._csr_ring_count_update.re,
                self._csr_ring_count_value.status.eq(self._ring_count)
            )
        ]
        
        # Helper signals for incrementing/decrementing _ring_count, and the statement that
        # actually updates _ring_count, or resets it when doing soft reset.
        self._ring_count_inc = Signal(8)
        self._ring_count_dec = Signal(1)
        self.sync += [
            If(self._do_soft_reset,
                self._ring_count.eq(0),
            ).Else(
                self._ring_count.eq(
                    self._ring_count + self._ring_count_inc - self._ring_count_dec)
            )
        ]

        # Implementation of the ring_submit CSR. If a value is being written to
        # csr_ring_submit then increment ring_count by the written value, no increment
        # is done.
        self.comb += [
            If(self._csr_ring_submit.re,
                self._ring_count_inc.eq(self._csr_ring_submit.storage)
            ).Else(
                self._ring_count_inc.eq(0)
            )
        ]
        
        # Implementation of releating a buffer and resetting _ring_pos. The FSM below is
        # responsible determining when to release a buffer by assigning to _release_buffer.
        # A buffer can only be released when _ring_pos is greater than 0.
        self._release_buffer = Signal(1)
        self.comb += [
            # Default value.
            self._release_buffer.eq(0),
            # Decrement ring_count by one when a buffer is released.
            self._ring_count_dec.eq(self._release_buffer)
        ]
        self.sync += [
            If(self._do_soft_reset,
                self._ring_pos.eq(0)
            )
            .Elif(self._release_buffer,
                If(self._ring_pos == self._csr_ring_size_m1.storage,
                    self._ring_pos.eq(0)
                ).Else(
                    self._ring_pos.eq(self._ring_pos + 1)
                )
            )
        ]
        
        # FSM which does the following in a loop:
        # - Wait for a buffer to be available (WAIT_BUFFER).
        # - Read the buffer descriptor (READ_DESC_1, READ_DESC_2).
        # - Decode the information from the buffer descriptor.
        # - Pass control to the derived class do do the DMA read/write and wait
        #   until it's done (PROCESS_BUFFER).
        # - If necessary, update the buffer descriptor (WRITE_DESC).
        # - Release the buffer.

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

        # Signals for temporary values. The _desc_value contains the read descriptor
        # value and is read by the derived class during PROCESS_BUFFER.
        desc_addr = Signal(wb_adr_width)
        self._desc_value = Signal(64)

        fsm.act("WAIT_BUFFER",
            # If soft reset is needed, do it.
            If(self._latch_soft_reset,
                # Comb-assign 1 to _do_soft_reset so that _ring_count and _ring_pos
                # will be reset to zero in next cycle transition.
                self._do_soft_reset.eq(1),
                # Clear _latch_soft_reset now that we have done the soft reset.
                NextValue(self._latch_soft_reset, 0)
            )
            # If we have a buffer, calculate the descriptor address and continue in
            # READ_DESC_1.
            .Elif(self._ring_count > 0,
                NextValue(desc_addr,
                    (self._csr_ring_addr.storage >> mem_align_addr_bits) +
                    self._ring_pos * desc_size_words),
                NextState("READ_DESC_1")
            )
        )

        fsm.act("READ_DESC_1",
            # Read the descriptor (first word or complete).
            wb_master.adr.eq(desc_addr),
            wb_master.sel.eq(0b1111 if wb_data_width == 32 else 0b11111111),
            wb_master.cyc.eq(1),
            wb_master.stb.eq(1),
            wb_master.we.eq(0),
            If(wb_master.err,
                NextState("ERROR")
            )
            .Elif(wb_master.ack,
                NextValue(self._desc_value[0:wb_data_width], wb_master.dat_r),
                If(wb_data_width == 32,
                    NextState("READ_DESC_INC_ADDR")
                ).Else(
                    NextState("PROCESS_BUFFER")
                )
            )
        )

        if wb_data_width == 32:
            fsm.act("READ_DESC_INC_ADDR",
                NextValue(desc_addr, desc_addr + 1),
                NextState("READ_DESC_2")
            )

            fsm.act("READ_DESC_2",
                wb_master.adr.eq(desc_addr),
                wb_master.sel.eq(0b1111),
                wb_master.cyc.eq(1),
                wb_master.stb.eq(1),
                wb_master.we.eq(0),
                If(wb_master.err,
                    NextState("ERROR")
                )
                .Elif(wb_master.ack,
                    NextValue(self._desc_value[wb_data_width:], wb_master.dat_r),
                    NextState("PROCESS_BUFFER")
                )
            )
        
        # The derived class indicates when processing of the buffer has been completed
        # by driving the _buffer_processed signal. It also indicates whether the
        # second word of the descriptor should be updated and to which value.
        self._buffer_processed = Signal(1)
        self._update_descriptor = Signal(1)
        self._update_descriptor_value = Signal(32)

        fsm.act("PROCESS_BUFFER",
            If(self._buffer_processed,
                If(self._update_descriptor,
                    NextState("WRITE_DESC")
                ).Else(
                    NextState("RELEASE_BUFFER")
                )
            )
        )

        fsm.act("WRITE_DESC",
            wb_master.adr.eq(desc_addr),
            wb_master.dat_w.eq(
                self._update_descriptor_value if wb_data_width == 32
                else (self._update_descriptor_value << 32)),
            wb_master.sel.eq(0b1111 if wb_data_width == 32 else 0b11110000),
            wb_master.cyc.eq(1),
            wb_master.stb.eq(1),
            wb_master.we.eq(1),
            If(wb_master.err,
                NextState("ERROR")
            )
            .Elif(wb_master.ack,
                NextState("RELEASE_BUFFER")
            )
        )

        fsm.act("RELEASE_BUFFER",
            self._release_buffer.eq(1),
            NextState("WAIT_BUFFER")
        )

        fsm.act("ERROR",
            # Stay here waiting for soft reset. Once found, just go to WAIT_BUFFER where
            # it will actually be handled.
            If(self._latch_soft_reset,
                NextState("WAIT_BUFFER")
            )
        )


class WishboneStreamDMARead(Module, AutoCSR, _WishboneStreamDMABase):
    def __init__(self, stream_desc, wb_data_width=32, wb_adr_width=30):
        _WishboneStreamDMABase.__init__(self, stream_desc, wb_data_width, wb_adr_width)

        self.source = stream.Endpoint(stream_desc)


class WishboneStreamDMAWrite(Module, AutoCSR, _WishboneStreamDMABase):
    def __init__(self, stream_desc, wb_data_width=32, wb_adr_width=30):
        _WishboneStreamDMABase.__init__(self, stream_desc, wb_data_width, wb_adr_width)

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

