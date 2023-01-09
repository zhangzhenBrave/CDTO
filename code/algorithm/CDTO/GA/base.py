from abc import abstractmethod

from util import create_colocation_groups, evaluate_placement
import multiprocessing


class BaseOptimizer:

    def __init__(self, colocation_heuristic=None, verbose=False, batches=1, pipeline_batches=1, n_threads=1,
                 score_save_period=None, simulator_comp_penalty=1, simulator_comm_penalty=1,
                 device_memory_utilization=1):
        self.colocation_heuristic = colocation_heuristic
        self.verbose = verbose
        self.batches = batches
        self.pipeline_batches = pipeline_batches
        self.simulator_comp_penalty = simulator_comp_penalty
        self.simulator_comm_penalty = simulator_comm_penalty
        self.device_memory_utilization = device_memory_utilization
        if n_threads == -1:
            self.n_threads = multiprocessing.cpu_count()
        else:
            self.n_threads = n_threads

        if score_save_period is None and verbose:
            self.score_save_period = verbose
        else:
            self.score_save_period = score_save_period

    def create_colocation_groups(self, layer_names):
        if not self.colocation_heuristic:
            return [[name] for name in layer_names]

        return create_colocation_groups(layer_names, self.colocation_heuristic)

    # def evaluate_placement(self, net, net_len, task_num,n_devices,task_iner_priority, device_graph, task_unit, batch_size=128):
    #     return evaluate_placement(net, task_iner_priority, device_graph, task_unit, batch_size=128, batches=1, pipeline_batches=1, memory_penalization_factor=1,
    #                    noise_std=0, comp_penalty=1, comm_penalty=1, device_memory_utilization=1)

    @abstractmethod
    def optimize(self, net_len, task_num,n_devices,task_iner_priority, device_graph, task_unit):
        raise NotImplementedError()
