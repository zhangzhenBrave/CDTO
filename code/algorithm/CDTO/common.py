
import numpy as np
import copy


import random
from flopsprofiler import FlopsProfiler
from algorithm import  ComputationGraph


def random_task(number,user):
    task_unit=int(number/user)

    list1 = {task: [] for task in range(0,user)}
    for i in range(0,user):
        for j in range(0,task_unit):
          list1[i].append(random.randint(0, 3))

    return list1

def calculate_tensor_size(shape, dtype='float32'):
    return np.prod(shape) * np.dtype(dtype).itemsize

def generate_diff_task(user,num,task_num,base_exc_time,net_graph_path1,net_graph_path2,net_graph_path3,net_graph_path4):
    task = []
    with open(net_graph_path1) as f:
        net_string1 = f.read()
    with open(net_graph_path2) as f:
        net_string2 = f.read()
    with open(net_graph_path3) as f:
        net_string3 = f.read()
    with open(net_graph_path4) as f:
        net_string4 = f.read()
    graph1 = ComputationGraph()
    graph1.load_from_string(net_string1)
    graph2 = ComputationGraph()
    graph2.load_from_string(net_string2)
    graph3 = ComputationGraph()
    graph3.load_from_string(net_string3)
    graph4 = ComputationGraph()
    graph4.load_from_string(net_string4)
    for i in range(0,user):
        for j in task_num[i]:
            if j==0:
                    task.append(copy.deepcopy(graph1))
            elif j==1:
                    task.append(copy.deepcopy(graph2))
            elif j==2:
                    task.append(copy.deepcopy(graph3))
            else:
                    task.append(copy.deepcopy(graph4))
    # print(np.sum(task_num))
    task_dl=[i*10+base_exc_time for i in range(num)]
    return task,task_dl

def generate_same_task(task_num,base_exc_time,net_graph_path1):
    task=[]
    with open(net_graph_path1) as f:
        net_string = f.read()
    graph = ComputationGraph()
    graph.load_from_string(net_string)
    for i in range(task_num[0]):
        task.append(copy.deepcopy(graph))
    task_dl = [random.random() * 10 + base_exc_time for i in range(np.sum(task_num))]
    return task, task_dl

def  task_sort(device,task, task_dl):
    task_iner_priority=[]
    peak_gflops=0
    for i in device.devices:
        peak_gflops+=i.device.peak_gflops
    aver_peak_gflops=peak_gflops/len(device.devices)
    band = 0
    for i in device.comm_channels:
        band += i.bandwidth
    aver_bandwidth = band  / len(device.comm_channels)/8
    task_priority = np.array(task)[np.argsort(np.array(task_dl))]
    # print(aver_peak_gflops)
    # print(aver_bandwidth)
    for graph in task_priority :
        task_iner_priority.append(iner_sorting(graph,aver_peak_gflops,aver_bandwidth))
    return task_iner_priority

def iner_sorting(graph,aver_peak_gflops,aver_bandwidth):

    incoming = {}
    for layer_spec in graph.topological_order:
        incoming[layer_spec] = 0
    for node in reversed(graph.topological_order):
        # print('node',node)
        # print(node.operation.outputs)
        d = calculate_tensor_size(node.operation.outputs, dtype='float32')
        gflops=FlopsProfiler.profile(node)
        # print(d/ 2 ** 30 /aver_bandwidth)
        # print(gflops/aver_peak_gflops)
        theta=(gflops/aver_peak_gflops)/(d/ 2 ** 30 /aver_bandwidth)
        # print('theta',theta)
        if 1-pow(1000,-theta)<np.random.rand():
          prob=0
        else:
          prob=1
        # print('prob',prob)
        max=0
        for m in node.outbounds:
               grade=incoming[m]+prob*(d/2 ** 30 /aver_bandwidth)
               if grade>=max:
                   max=grade
        incoming[node]=max+gflops/aver_peak_gflops
        # print('grade',incoming[node])
    a1 = sorted(incoming.items(), key=lambda x: x[1],reverse=True)
    task_iner_priority = [ k for k, v in dict(a1).items()]
    return task_iner_priority

