from paleo.profilers.flops_profiler import FlopsProfiler as PaleoFlopsProfiler
from paleo.profilers.base import ProfilerOptions


class FlopsProfiler:
    @staticmethod
    def profile(layer_spec):
        layer = layer_spec.operation

        assert layer is not None, f'{layer_spec} has no operation'


        profiler_options = ProfilerOptions()
        direction = 'forward'
        profiler_options.direction = direction
        profiler_options.use_cudnn_heuristics = False
        profiler_options.include_bias_and_activation = False
        class device:
            def __init__(self,type):
                self.type = type
            def is_gpu(self):
                return self.type == 'cpu'
        profiler = PaleoFlopsProfiler(profiler_options, device('cpu'))
        gflops = profiler.flop_profile(layer)
        # print('time',time)

        return gflops


