import numpy as np

from paleo.profilers.flops_profiler import FlopsProfiler as PaleoFlopsProfiler
from paleo.profilers.base import ProfilerOptions


def calculate_tensor_size(shape, dtype='float32'):
    return np.prod(shape) * np.dtype(dtype).itemsize


class TransferProfiler:
    @staticmethod
    def profile(layer_spec, comm_channel,hop, parent_device,child_device,parent_relate_comm_channel=None,child_relate_comm_channel=None, backward=False, batch_size=10, dtype='float32',
                comm_penalization=1, comp_penalization=1):
        layer = layer_spec.operation

        # if batch_size:
        #     layer.batch_size = batch_size

        profiler_options = ProfilerOptions()
        direction = 'backward' if backward else 'forward'
        profiler_options.direction = direction
        profiler_options.use_cudnn_heuristics = False
        profiler_options.include_bias_and_activation = False
        profiler_options.ppp_comp = comp_penalization
        profiler_options.ppp_comm = comm_penalization

        profiler = PaleoFlopsProfiler(profiler_options, parent_device.device)
        # print(hop)

        num_bytes = calculate_tensor_size(layer.outputs, dtype)
        if parent_device.device.type=='cpu' and child_device.device.type!='cpu':
            # print(1)
            # print(comm_channel.bandwidth )
            # print(parent_relate_comm_channel.bandwidth)
            time = profiler.cpu1estimate_comm_time(
                num_bytes,hop, comm_channel.bandwidth / 8,parent_relate_comm_channel.bandwidth/ 8, ppp=profiler.options.ppp_comm)
        elif parent_device.device.type!='cpu' and child_device.device.type=='cpu':
            time = profiler.cpu2estimate_comm_time(
                num_bytes,hop, comm_channel.bandwidth / 8,child_relate_comm_channel.bandwidth/ 8,  ppp=profiler.options.ppp_comm)
            # print(2)
            # print(comm_channel.bandwidth)
            # print(child_relate_comm_channel.bandwidth)
        elif parent_device.device.type == 'cpu' and child_device.device.type == 'cpu':
            # print(2)
            # print('22',comm_channel.bandwidth)
            # print('23',parent_relate_comm_channel.bandwidth)
            # print('24',child_relate_comm_channel.bandwidth)
            time = profiler.cpu3estimate_comm_time(
                num_bytes, hop, comm_channel.bandwidth / 8,parent_relate_comm_channel.bandwidth/ 8, child_relate_comm_channel.bandwidth / 8, ppp=profiler.options.ppp_comm)
        else:
            time = profiler.estimate_comm_time(
                num_bytes * hop, comm_channel.bandwidth / 8, ppp=profiler.options.ppp_comm)

        return time
