"""
Execution simulator based on the technique described in Placeto (Addanki et al., 2019, https://arxiv.org/abs/1906.08879)
"""

import heapq


import numpy as np


from algorithm.profilers import FlopsProfiler
from algorithm.profilers import TransferProfiler


class Simulator_local_cloud:

    def __init__(self, task_iner_priority, device_graph,task_unit):
        self.device_graph = device_graph
        self.task_iner_priority = task_iner_priority
        self.transfer_device_run_time = [[] for task in range(len(self.task_iner_priority))]
        self.op_device_run_time = {task: {} for task in range(len(self.task_iner_priority))}
        self.device_finishtime = [0 for device in self.device_graph.devices]
        self.transfer_finishtime = [0 for comm_channel in self.device_graph.comm_channels]
        self.t = list(map(int, np.zeros(len(self.task_iner_priority))))
        self.last_reward = 0
        self.device_gflop = self.computing_capacity()
        self.batch_size = 1
        self.comp_penalization = 1
        self.comm_penalization = 1
        self.task_finish=[]
        self.task_unit = task_unit


    def run_op(self, op, Device, start_time):
        device = self.device_graph.devices[Device].device
        run_time = FlopsProfiler.profile(op, device, False, self.batch_size, comp_penalization=self.comp_penalization,
                                         comm_penalization=self.comm_penalization)
        end_time = start_time + run_time

        return end_time

    def run_transfer(self, op, fromdevice, todevice, comm_channel, hop, start_time):
        parent_device = self.device_graph.devices[fromdevice]
        child_device = self.device_graph.devices[todevice]
        parent_relate_comm_channel = None
        child_relate_comm_channel = None
        if parent_device.device.type == "cpu":
            parent_relate_device = self.device_graph.devices[parent_device.device.relate]
            parent_relate_comm_channel = parent_relate_device.neighbours[child_device]
        if child_device.device.type == "cpu":
            child_relate_device = self.device_graph.devices[child_device.device.relate]
            child_relate_comm_channel = child_device.neighbours[child_relate_device]

        transfer_time = TransferProfiler.profile(op, comm_channel, hop, parent_device, child_device,
                                                 parent_relate_comm_channel, child_relate_comm_channel, False,
                                                 self.batch_size, comm_penalization=self.comm_penalization,
                                                 comp_penalization=self.comp_penalization)
        end_time = start_time + transfer_time
        return end_time
    def simulate(self):
        split=0.1
        for q in range(127):
            for j  in range(len(self.task_iner_priority)):

                i = int(j / self.task_unit)
                if self.t[j] >= len(self.task_iner_priority[j]) :
                    continue
                layer = self.task_iner_priority[j][self.t[j]]
                if self.t[j] <=len(self.task_iner_priority[j])*split:
                    device = i
                else:
                    device=len(self.device_graph.devices)-1
                if layer.name == 'output' or layer.name == 'data' :
                    device=i
                    layer.device_id = device
                    layer_end_time = self.update_time(j, device, layer)
                    if layer.name == 'output':
                        self.task_finish.append(layer_end_time)
                else:

                    layer_end_time = self.update_time(j,device, layer)
                    layer.device_id = device


                self.t[j] = self.t[j] + 1
        return  self.task_finish

    def update_time(self, i, device, layer):

        max_pd = 0
        for parent in layer.inbounds:
            if parent.device_id != device:


                finish_trans_tensor = next((m for m in self.transfer_device_run_time[i]
                                            if m[:3] == [parent, parent.device_id, device]), None)

                if finish_trans_tensor == None:
                    op_device = self.device_graph.devices[parent.device_id]
                    child_device = self.device_graph.devices[device]
                    comm_channel = op_device.neighbours[child_device]
                    hop = op_device.neighbourshop[child_device]

                    trans_start_time = max(self.op_device_run_time[i][parent][2],
                                           self.transfer_finishtime[comm_channel.id])
                    trans_end_time = self.run_transfer(parent, parent.device_id, device, comm_channel, hop,
                                                       trans_start_time)
                    self.transfer_finishtime[comm_channel.id] = trans_end_time
                    self.transfer_device_run_time[i].append([parent, parent.device_id, device,
                                                             trans_start_time,
                                                             trans_end_time])
                    # print('transfer', [parent, parent.device_id, device,
                    #                    trans_start_time, trans_end_time])
                    if trans_end_time >= max_pd:
                        max_pd = trans_end_time
                else:
                    # print(1000000000000000000)
                    if finish_trans_tensor[-1] >= max_pd:
                        max_pd = finish_trans_tensor[-1]
            else:
                if self.op_device_run_time[i][parent][2] >= max_pd:
                    max_pd = self.op_device_run_time[i][parent][2]
        # print('self.device_finishtime[device]', self.device_finishtime[device])
        layer_start_time = max(max_pd, self.device_finishtime[device])
        layer_end_time = self.run_op(layer, device, layer_start_time)


        self.op_device_run_time[i][layer] = (device, layer_start_time, layer_end_time)
        # print(self.op_device_run_time[i][layer])
        self.device_finishtime[device] = layer_end_time
        return layer_end_time
    def computing_capacity(self):
        device_gflop = []
        for i in range(len(self.device_graph.devices)):
            device_gflop.append(self.device_graph.devices[i].device.peak_gflops)
        return device_gflop

    def update_time1(self,j, i,  layer):
        # print('layer.name', layer.name)
        # print('layer.outputs', layer.outputs)
        min = 1000
        last_device = 0
        last_layer_start_time = 0
        # 在多个设备中选出完成时间最少的
        devices=[i,len(self.device_graph.devices)-1]

        for device in devices:
            # print('device',device)
            max_pd = 0
            for parent in layer.inbounds:
                if parent.device_id != device:

                    finish_trans_tensor = next((m for m in self.transfer_device_run_time[j]
                                                if m[:3] == [parent, parent.device_id, device]), None)
                    # print('finish_trans_tensor',finish_trans_tensor)
                    if finish_trans_tensor == None:
                        op_device = self.device_graph.devices[parent.device_id]
                        child_device = self.device_graph.devices[device]
                        comm_channel = op_device.neighbours[child_device]
                        # print(comm_channel)
                        hop = op_device.neighbourshop[child_device]
                        trans_start_time = max(self.op_device_run_time[j][parent][2],
                                               self.transfer_finishtime[comm_channel.id])
                        # print((parent.device_id,device))
                        trans_end_time = self.run_transfer(parent, parent.device_id, device, comm_channel, hop,
                                                           trans_start_time)

                        # add

                        if trans_end_time >= max_pd:
                            max_pd = trans_end_time
                    else:
                        if finish_trans_tensor[-1] >= max_pd:
                            max_pd = finish_trans_tensor[-1]
                else:
                    if self.op_device_run_time[j][parent][2] >= max_pd:
                        max_pd = self.op_device_run_time[j][parent][2]

            layer_start_time = max(max_pd, self.device_finishtime[device])
            layer_end_time = self.run_op(layer, device, layer_start_time)
            # if layer.name == 'Mixed_6c/Branch_2/Conv2d_0e_1x7'and i==1:
            #
            #     print('layer_start_time',layer_start_time)
            #     print('layer_end_time',layer_end_time)
            # add
            if (min >= layer_end_time):
                min = layer_end_time
                last_device = device
                last_layer_start_time = layer_start_time

        layer_end_time = self.update_time(j, last_device, layer)

        return layer_end_time, last_device
    def neighbor_hop(self, device):
        hop = []
        op_device = self.device_graph.devices[device]
        for i in op_device.neighbours:
            hop.append(op_device.neighbourshop[i])
        return hop





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