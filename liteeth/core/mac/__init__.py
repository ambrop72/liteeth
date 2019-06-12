from liteeth.common import *
from liteeth.core.mac.common import *
from liteeth.core.mac.core import LiteEthMACCore
from liteeth.core.mac.wishbone import LiteEthMACWishboneInterface
from liteeth.core.mac.wishbone_dma import LiteEthMACWishboneDMA


class LiteEthMAC(Module, AutoCSR):
    def __init__(self, phy, dw,
                 interface="crossbar",
                 endianness="big",
                 with_preamble_crc=True,
                 nrxslots=2,
                 ntxslots=2):
        self.submodules.core = LiteEthMACCore(phy, dw, endianness, with_preamble_crc)
        self.csrs = []
        if interface == "crossbar":
            self.submodules.crossbar = LiteEthMACCrossbar()
            self.submodules.packetizer = LiteEthMACPacketizer()
            self.submodules.depacketizer = LiteEthMACDepacketizer()
            self.comb += [
                self.crossbar.master.source.connect(self.packetizer.sink),
                self.packetizer.source.connect(self.core.sink),
                self.core.source.connect(self.depacketizer.sink),
                self.depacketizer.source.connect(self.crossbar.master.sink)
            ]
        elif interface == "wishbone":
            self.rx_slots = CSRConstant(nrxslots)
            self.tx_slots = CSRConstant(ntxslots)
            self.slot_size = CSRConstant(2**bits_for(eth_mtu))
            self.submodules.interface = LiteEthMACWishboneInterface(dw, nrxslots, ntxslots, endianness)
            self.comb += Port.connect(self.interface, self.core)
            self.ev, self.bus = self.interface.sram.ev, self.interface.bus
            self.csrs = self.interface.get_csrs() + self.core.get_csrs()
        elif interface == "wishbone_dma":
            self.submodules.wishbone_dma = LiteEthMACWishboneDMA(dw)
            self.comb += Port.connect(self.wishbone_dma, self.core)
            self.wb_master_tx = self.wishbone_dma.wb_master_tx
            self.wb_master_rx = self.wishbone_dma.wb_master_rx
            self.ev = self.wishbone_dma.ev
            self.csrs = self.wishbone_dma.get_csrs() + self.core.get_csrs()
        else:
            raise NotImplementedError

    def get_csrs(self):
        return self.csrs
