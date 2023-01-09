
import argparse
import random
import numpy as np
from algorithm import DeviceGraph
import time
from algorithm.CDTO.common import generate_diff_task,generate_same_task,task_sort,random_task
from algorithm.CDTO.cdto import  Simulator_greedy
from algorithm.CDTO.base_local import  Simulator_local
from algorithm.CDTO.base_cloud import  Simulator_cloud
from algorithm.CDTO.local_cloud import  Simulator_local_cloud
from algorithm.CDTO.base_ergodic import  Simulator_ergodic
from genetic_algorithm import GAOptimizer
def four_schemes(parse):

    local_time=[]
    cloud_time=[]
    cdto_time=[]
    local_cloud_time=[]
    ergodic_time=[]
    user=10
    time_start = time.clock()  # 记录开始时间
    for i in range(1):
        task=random_task(parse.num,user)
        print(task)
        task_unit=int(parse.num/user)

        #print(task_num)
        device_graph = DeviceGraph.load_from_file( parse.device_graph_path)
        if parse.diff_task:#是否是相同任务
            task, task_dl=generate_diff_task(user,parse.num,task,parse.task_release_time,parse.net_graph_path1,parse.net_graph_path2,parse.net_graph_path3,parse.net_graph_path4)
        else:
            task, task_dl=generate_same_task(user,parse.num,task,parse.task_release_time,parse.net_graph_path1)

        task_iner_priority=task_sort(device_graph,task, task_dl)
        task_dim=[]
        time_end1 = time.clock()  # 记录结束时间
        time_sum1= time_end1- time_start  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum1)
        ##local scheme#######################################################

        simulator1 = Simulator_local(task_iner_priority, device_graph,task_unit)
        task_finish_time2=simulator1.simulate()
        # print(task_finish_time2)
        local_time.append(max(task_finish_time2))
        print("##############################")
        time_end2 = time.clock()  # 记录结束时间
        time_sum2= time_end2- time_end1  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum2)

        #cloud  scheme#######################################################
        simulator2 = Simulator_cloud(task_iner_priority, device_graph,task_unit)
        for i in range(127):
            # print(i)
            if i==0 or i==126:
                device = [int(task/ task_unit) for task in range(len(task_iner_priority))]

            else:
                 # print(i)
                 device = [len(device_graph.devices)-1 for task in range(len(task_iner_priority))]

                 # device = devices[i-1]
            task_finish_time3=simulator2.simulate(device)

        cloud_time.append(max(task_finish_time3))
        print("##############################")
        time_end3 = time.clock()  # 记录结束时间
        time_sum3= time_end3- time_end2  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum3)

        #CDTO scheme#######################################################
        simulator3 = Simulator_greedy(task_iner_priority, device_graph,task_unit)
        task_finish_time4 = simulator3.simulate()
        # print(task_finish_time4)
        cdto_time.append(max(task_finish_time4))
        time_end4 = time.clock()  # 记录结束时间
        time_sum4= time_end4- time_end3  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum4)
        print("##############################")
        #local_cloud scheme#######################################################
        simulator4 = Simulator_local_cloud(task_iner_priority, device_graph,task_unit)
        task_finish_time5 = simulator4.simulate()
        local_cloud_time.append(max(task_finish_time5))
        time_end5 = time.clock()  # 记录结束时间
        time_sum5= time_end5- time_end4  # 计算的时间差为程序的执行时间，单位为秒/s
        print(time_sum5)
         ########################################################

        #ergodic  scheme#######################################################
        optimizer = GAOptimizer(plot_fitness_history=True,
                                generations= 2000,
                                population_size= 100,
                                mutation_rate=0.8,
                                zone_mutation_rate=0.2,
                                mutation_sharding_rate=0,
                                crossover_rate=0.8,
                                crossover_type="1-point",
                                parent_selection_mechanism="rank",
                                evolve_mutation_rate=True,
                                verbose=5,
                                elite_size=5,
                                max_mutation_rate=0.9,
                                min_mutation_rate=0.05,
                                print_diversity=True,
                                include_trivial_solutions_in_initialization=False,
                                allow_cpu= True,
                                pipeline_batches=2,
                                batches=10,
                                n_threads=-1,
                                checkpoint_period=5,
                                simulator_comp_penalty=0.9,
                                simulator_comm_penalty=0.25)
        best_solution, best_placement =optimizer.optimize(net_len=127-2,
                                        task_num=parse.num,
                                        n_devices=len(device_graph.devices),
                                        task_iner_priority=task_iner_priority,
                                        device_graph=device_graph,
                                        task_unit=task_unit)
        print(best_solution)
        print(best_placement)


        print("##############################")



    print('local_time',np.average(local_time))
    print('cloud_time',np.average(cloud_time))
    print('local_cloud_time', np.average(local_cloud_time))
    print('network_time', np.average(cdto_time))
    # # print('ergodic_time1', ergodic_time)
    # print('ergodic_time', min(ergodic_time))
if __name__ == '__main__':
    np.random.seed(3)
    random.seed(3)
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_graph_path", type=str, default='../../configs/device_graphs/arpanet.json')
    parser.add_argument("--diff_task", type=bool, default=True)
    parser.add_argument("--num", type=list, default=250)
    parser.add_argument("--task_release_time", type=int, default=500,help='ms')
    parser.add_argument("--net_graph_path1", type=str, default='../../nets/alex_v2.json')
    parser.add_argument("--net_graph_path2", type=str, default='../../nets/inception_v3.json')
    parser.add_argument("--net_graph_path3", type=str, default='../../nets/resnet34.json')
    parser.add_argument("--net_graph_path4", type=str, default='../../nets/vgg16.json')

    args = parser.parse_args()

    four_schemes(args)









