import numpy as np
import pickle

def parse_Q(file="/Users/musk/Desktop/H_0_5500.pkl"):

    with open(file, "rb") as f:
        coe = pickle.load(f)

    UE_num = 20
    BS_num = 2
    Q_dict = dict()
    Q = np.zeros([UE_num * BS_num, UE_num * BS_num])

    for j in range(BS_num):
        for i in range(UE_num):
            Q_dict["X[{}][{}]".format(i, j)] = UE_num * j + i

    key_list = coe.keys()
    for key in key_list:
        key_i, key_j = key
        i = Q_dict[key_i]
        j = Q_dict[key_j]
        Q[i, j] = Q[j, i] = coe[key] / 2

    return Q

Q = parse_Q()

