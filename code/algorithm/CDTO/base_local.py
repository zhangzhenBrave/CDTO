"""
Execution simulator based on the technique described in Placeto (Addanki et al., 2019, https://arxiv.org/abs/1906.08879)
"""

import heapq
from collections import deque

import numpy as np

from algorithm.profilers import FlopsProfiler
from algorithm.profilers import TransferProfiler


class Simulator_local:

    def __init__(self, task_iner_priority, device_graph,task_unit):
        self.device_graph = device_graph
        self.task_iner_priority = task_iner_priority
        self.task_unit = task_unit


    def simulate(self,batch_size=1,comp_penalization=1, comm_penalization=1):
        op_queues = [deque() for device in self.device_graph.devices]

        op_run_time={task:{} for task in range(len(self.task_iner_priority))}
        device_finishtime = [0 for device in self.device_graph.devices]
        self.memory_usage = np.zeros(len(self.device_graph.devices))
        self.saved_tensors = [[] for i in range(len(self.device_graph.devices))]
        task_finish = []

        def run_op(op, Device, start_time):
            device = self.device_graph.devices[Device].device
            run_time = FlopsProfiler.profile(op, device, False, batch_size, comp_penalization=comp_penalization,
                                             comm_penalization=comm_penalization)
            end_time = start_time + run_time

            return end_time


        for j,task in enumerate(self.task_iner_priority):
            i=int(j/self.task_unit)
            for layer in task:
                if layer.name=='data':
                    op_queues[i].append((layer, False))
                    start_time=device_finishtime[i]
                    end_time=run_op(layer,i, start_time)

                    op_run_time[i][layer] = (i, start_time, end_time)
                    device_finishtime[i] = end_time
                else:
                    device=i
                    max_pd=0
                    self.memory_usage[i] += layer.operation.weights_in_bytes
                    for  parent  in layer.inbounds:
                         if op_run_time[i][parent][2] >= max_pd:
                             max_pd = op_run_time[i][parent][2]
                    layer_start_time=max(max_pd,device_finishtime[device])
                    layer_end_time = run_op(layer,device, layer_start_time)


                    op_run_time[i][layer] = (device, layer_start_time,layer_end_time)
                    if layer.name=='output':
                        task_finish.append(layer_end_time)
                    device_finishtime[device] = layer_end_time
                    op_queues[device].append((layer, False))
        return task_finish



class MinHeap:
    def __init__(self, initial=None):
        if initial:
            self._data = initial[:]
            heapq.heapify(self._data)
        else:
            self._data = []

    def push(self, item):
        heapq.heappush(self._data, item)

    def pop(self):
        return heapq.heappop(self._data)

    def empty(self):
        return len(self._data) == 0

    def __iter__(self):
        return self._data.__iter__()


class Event:

    def __init__(self, event_type, device,  operation=None, subtype=None,
                 from_device=None, to_device=None):
        self.type = event_type
        self.device = device
        self.operation = operation
        self.subtype = subtype
        self.from_device = from_device
        self.to_device = to_device

        self.handled = False
