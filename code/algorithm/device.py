from enum import Enum
import json


class DeviceType(Enum):
    GPU = 'gpu'
    CPU = 'cpu'


class CommunicationType(Enum):
    ETHERNET = 'ethernet'
    INFINIBAND = 'infiniband'
    PCIE3 = 'pcie3'
    SLI = 'sli'


class Device:

    def __init__(self, model, clock, peak_gflops, memory,relate=None, mem_bandwidth=68, type=DeviceType.CPU, hardware_id=None):
        """
        Creates a new device.
        :param model: Model string for the device. E.g. 'V100' or 'Titan X'.
        :param clock: Clock speed of the device, in MHz.
        :param peak_gflops: Peak GFLOPS (Floating Operations Per Second) of the device
        :param memory: Memory available to the device, in GB.
        :param mem_bandwidth: The bandwidth available to the device when reading from device memory, in GB/s.
        :param type: Device type; available types are given in the DeviceType enum. (CPU or GPU)
        :param hardware_id: The hardware ID of the device if available on the local computer.
        """

        self.model = model
        self.clock = clock
        self.peak_gflops = peak_gflops
        self.memory = memory
        self.mem_bandwidth = mem_bandwidth
        self.type = type
        self.hardware_id = hardware_id
        self.relate=relate


    @property
    def is_gpu(self):
        return self.type == DeviceType.GPU


class CommunicationChannel:

    def __init__(self, type, bandwidth, cid):
        """
        Creates a new communication channel.
        :param type:    The type of the channel; available types are given in the CommunicationType enum.
                        E.g. ethernet or Infiniband.
        :param bandwidth: The bandwidth of the channel, given in Gb/s.
        """
        self.type = type
        self.bandwidth = bandwidth
        self.id = cid


class DeviceNode:

    def __init__(self, device, device_id):
        self.device = device
        self.neighbours = {}
        self.neighbourshop = {}
        self.id = device_id


    def add_neighbour(self, device_node, comm_channel,hop ):
        # We save neighbours as a mapping to comm_channel, so that it is easy to find bandwidth
        self.neighbours[device_node] = comm_channel
        self.neighbourshop[device_node] = hop
    @property
    def name(self):
        if self.device.hardware_id:
            return self.device.hardware_id
        return 'device' + str(self.id)


class DeviceGraph:

    def __init__(self):
        self.devices = []
        self.comm_channels = []
        self.all_devices = []

    @staticmethod
    def load_from_file(path):
        device_graph = DeviceGraph()
        with open(path) as f:
            graph = json.loads(f.read())

            # First, we load all devices
            for i, device in enumerate(graph['devices']):
                if device['type']=='cpu':
                  args = [device['model'], device['clock'], device['peak_gflops'], device['memory'],device['relate']]
                else:
                    args = [device['model'], device['clock'], device['peak_gflops'], device['memory']]
                kwargs = {}

                for kw in ('mem_bandwidth', 'type', 'id'):
                    if kw in device:
                        kwargs[kw] = device[kw]

                d = Device(*args, **kwargs)
                device_graph.devices.append(DeviceNode(d, i))
            device_graph.all_devices.extend(device_graph.devices)

            # Then, we load all communication channels
            for i, comm_channel in enumerate(graph['comm_channels']):
                device_graph.comm_channels.append(CommunicationChannel(comm_channel['type'],
                                                                       comm_channel['bandwidth'], i))
            device_graph.all_devices.extend(device_graph.comm_channels)

            # Finally, we resolve neighbours
            for i, device in enumerate(device_graph.devices):
                # print(i)

                json_device = graph['devices'][i]

                for neighbour in json_device['neighbours']:
                    # print(neighbour)
                    comm_channel_idx = neighbour['comm_channel']
                    hop=neighbour['hop']
                    comm_channel = device_graph.comm_channels[comm_channel_idx]
                    # print((neighbour['device'], comm_channel, hop))

                    device.add_neighbour(device_graph.devices[neighbour['device']], comm_channel , hop)
        # print('device_graph',device_graph)
        return device_graph
