import collections
import json
from random import randint
import random
import numpy as np
def apply_placement(task_dim,placement):
    # print("##############################")
    # print(placement)
    print(task_dim)
    print(max(task_dim))
    net=np.ones((max(task_dim)-2,len(task_dim)), dtype=int)
    print(net)
    j=0
    for i in range(len(task_dim)):
        print(net[:task_dim[i],i])
        net[:task_dim[i]-2,i]    =placement[j:j+task_dim[i]-2]
        j+=task_dim[i]-2
    print(net)
    #
    # net=np.array(placement).reshape(net_len, task_num).tolist()


    return net
task_dim=[10,8]
placement=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
a=apply_placement(task_dim,placement)
# print(a)