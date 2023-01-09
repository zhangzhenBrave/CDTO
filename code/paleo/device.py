"""Spec of devices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Device(object):
    """Specification for devices."""

    def __init__(self, name, clock, peak_gflops, mem_bandwidth, is_gpu=False):
        """
        Args:
            name: device name
            clock: MHz
            peak_gflops: GFLOPS
            mem_bandwidth: GB/sec
        """
        self._name = name
        self.clock = clock
        self.peak_gflops = float(peak_gflops)
        self.mem_bandwidth = float(mem_bandwidth)
        self._is_gpu = is_gpu

    @property
    def name(self):
        return self._name

    @property
    def is_gpu(self):
        return self._is_gpu


_Gbps = 1
_GBps = 8


class Network(object):
    def __init__(self, name, bandwidth):
        """
        Args:
            name: name of this network connection.
            bandwidth: in Gbps.
        """
        self._name = name
        self._bandwidth_Gbps = bandwidth

    @property
    def name(self):
        return self._name

    @property
    def bandwidth(self):
        return self._bandwidth_Gbps


AWS = Network('AWS', bandwidth=2 * _Gbps)
ETHERNET = Network('Ethernet', bandwidth=10 * _Gbps)
ETHERNET_20 = Network('Ethernet', bandwidth=20 * _Gbps)
INFINIBAND = Network('Infiniband', bandwidth=70 * _Gbps)
PCIe_2 = Network('PCIe 2.0', bandwidth=8 * _GBps)  # One weird trick: 6 GB/s
PCIe_3 = Network('PCIe 3.0', bandwidth=16 * _GBps)

# Here we assume PCIe x16.
# PCIe 1.0: 150MB/s per lane per direction.
# PCIe 2.0: 500MB/s per lane per direction.
# PCIe 3.0: 1GB/s per lane per direction.

NETWORKS = {'aws': AWS,
            'ethernet': ETHERNET,
            'ethernet20': ETHERNET_20,
            'infiniband': INFINIBAND,
            'pcie2': PCIe_2,
            'pcie3': PCIe_3}

# Predefined devices.
GPU_TITAN_X = Device(
    'Titan X', clock=1000, peak_gflops=6144, mem_bandwidth=336.5, is_gpu=True)

GPU_K20 = Device(
    'K20', clock=1000, peak_gflops=3520, mem_bandwidth=208, is_gpu=True)

GPU_K20X = Device(
    'K20X', clock=1000, peak_gflops=3935, mem_bandwidth=250, is_gpu=True)

GPU_K40 = Device(
    'K40', clock=745, peak_gflops=4290, mem_bandwidth=288, is_gpu=True)

GPU_K80 = Device(
    'K80', clock=560, peak_gflops=5600, mem_bandwidth=480, is_gpu=True)

# Note: Peak FLOPS for tensors is 125GFLOPS! Perhaps the model is too simple for specialized hardware?
GPU_V100 = Device(
    'V100', clock=1530, peak_gflops=15700, mem_bandwidth=900, is_gpu=True
)

GPU_GEFORCE_780_TI = Device(
    'GeForce 780 Ti',
    clock=875,
    peak_gflops=5040,
    mem_bandwidth=336,
    is_gpu=True)

GPU_GEFORCE_750M = Device(
    'GeForce 750 M',
    clock=941,
    peak_gflops=722.7,
    mem_bandwidth=80,
    is_gpu=True)

CPU_I7_5930K = Device(
    'CPU i7 5930K', clock=6 * 35000, peak_gflops=289, mem_bandwidth=68)

CPU_I7_4770K = Device(
    'CPU i7 4770K', clock=4 * 34000, peak_gflops=206, mem_bandwidth=68)

DEVICES = {
    'TITAN_X': GPU_TITAN_X,
    'K20': GPU_K20,
    'K20X': GPU_K20X,
    'K40': GPU_K40,
    'K80': GPU_K80,
    'V100': GPU_V100,
    'GEFORCE_780_TI': GPU_GEFORCE_780_TI,
    'GEFORCE_750_M': GPU_GEFORCE_750M,
    'CPU_I7': CPU_I7_5930K,
    'CPU_I7_4770K': CPU_I7_4770K,
}
