import numpy as np
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from numba import njit
import time

# np.random.seed(1)

def symmetric(n):
    sym = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            sym[i, j] = sym[j, i] = np.random.rand()-0.5
    return sym

def get_annealing_beta(beta_range, num_sweeps):
    beta_start, beta_stop = beta_range
    beta = np.geomspace(beta_start, beta_stop, num_sweeps)
    return beta

def slip(b, Q, offset, beta, n=None):
    delta_E = -2*2*b[n]*(np.sum(b*Q[n])-b[n]*Q[n, n])-offset
    p = np.exp(-delta_E*beta)
    if delta_E<0:
        accept = True
    elif p > np.random.rand():
        accept = True
    else:
        accept = False
    return n, accept, delta_E

@njit
def slip_binary(b, Q, offset, beta, n=None):
    b_tmp = np.copy(b)
    if b_tmp[n] == 1:
        delta_E = -2*np.sum(b_tmp*Q[n])-offset
    else:
        b_tmp[n] = 1
        delta_E = 2*np.sum(b_tmp*Q[n])-offset

    p = np.exp(-delta_E*beta)
    if delta_E<0:
        accept = True
    elif p > np.random.rand():
        accept = True
    else:
        accept = False
    return n, accept, delta_E

def DigitalAnnealing(b, Q, num_sweeps=1000, beta_range=(1, 50)):
    """
    a run of digital annealing algorithm on CPU

    :param b: inititial binary(spin) state
    :param Q: Q matrix
    :param num_sweeps: number of iteration in an annealing process (a run)
    :param beta_range: 1/T, beta is rising up form low to high means temperature is cooling
    :return: the state has minimum energy and its energy
    """
    beta = get_annealing_beta(beta_range=beta_range, num_sweeps=num_sweeps)
    offset = 0
    offset_increasing_rate = 0.1
    var_num = len(b)

    delta_E = np.zeros([var_num])
    e = np.zeros([num_sweeps])
    for i in tqdm(range(num_sweeps)):
        flip_list = []
        for qi in range(var_num):
            n, accept, delta_E[qi] = slip_binary(b, Q, offset, beta[i], qi)
            if accept :
                flip_list.append(n)

        if len(flip_list) == 0:
            offset += offset_increasing_rate*np.min(delta_E)
        else:
            # b[np.random.choice(flip_list)] *= -1
            rId = np.random.choice(flip_list)
            b[rId] = b[rId] * -1 + 1
            offset = 0

        e[i] = np.matmul(np.matmul(b.T, Q), b)

    # e_min = np.matmul(np.matmul(b.T, Q), b)
    return b, e

def DigitalAnnealing_multiThread(b, Q, num_sweeps=1000, beta_range=(1, 50), thread_num=3):
    """
    a run of digital annealing algorithm on CPU

    :param b: inititial binary(spin) state
    :param Q: Q matrix
    :param num_sweeps: number of iteration in an annealing process (a run)
    :param beta_range: 1/T, beta is rising up form low to high means temperature is cooling
    :return: the state has minimum energy and its energy
    """
    beta = get_annealing_beta(beta_range=beta_range, num_sweeps=num_sweeps)
    offset = 0
    offset_increasing_rate = 0.1
    var_num = len(b)

    delta_E = np.zeros([var_num])
    e = np.zeros([num_sweeps])

    pool = Pool(thread_num)

    for i in tqdm(range(num_sweeps)):

        func = partial(slip_binary, b, Q, offset, beta[i])
        all_step = np.vstack(pool.map(func, range(var_num)))
        accept = np.where(all_step[:, 1]==1)

        if len(accept) == 0:
            offset += offset_increasing_rate*np.min(all_step[:, 2])
        else:
            index = np.random.choice(accept[0])
            # b[index] *= -1
            b[index] = b[index] * -1 + 1
            offset = 0

        e[i] = np.matmul(np.matmul(b.T, Q), b)

    # e_min = np.matmul(np.matmul(b.T, Q), b)
    return b, e

@njit
def DigitalAnnealing_numba(b, Q, num_sweeps=1000, beta_range=(1, 50), beta=None):
    """
    a run of digital annealing algorithm on CPU

    :param b: inititial binary(spin) state
    :param Q: Q matrix
    :param num_sweeps: number of iteration in an annealing process (a run)
    :param beta_range: 1/T, beta is rising up form low to high means temperature is cooling
    :return: the state has minimum energy and its energy
    """
    # beta_start, beta_stop = beta_range
    # beta = np.geomspace(beta_start, beta_stop, num_sweeps)
    offset = 0
    offset_increasing_rate = 0.1
    var_num = len(b)

    delta_E = np.zeros(var_num)
    e = np.zeros(num_sweeps)
    for i in range(num_sweeps):
        flip_list = []
        for qi in range(var_num):
            b_tmp = np.copy(b)
            if b_tmp[qi] == 1:
                delta_E[qi] = -2 * np.sum(b_tmp * Q[qi]) - offset
            else:
                b_tmp[qi] = 1
                delta_E[qi] = 2 * np.sum(b_tmp * Q[qi]) - offset

            p = np.exp(-delta_E[qi] * beta[qi])
            if delta_E[qi] < 0:
                accept = True
            elif p > np.random.rand():
                accept = True
            else:
                accept = False
            # n, accept, delta_E[qi] = slip_binary(b, Q, offset, beta[i], qi)
            if accept :
                flip_list.append(qi)

        if len(flip_list) == 0:
            offset += offset_increasing_rate*np.min(delta_E)
        else:
            # b[np.random.choice(flip_list)] *= -1
            b[np.random.choice(np.array(flip_list))] = b[np.random.choice(np.array(flip_list))] * -1 + 1
            offset = 0

        # e[i] = np.matmul(np.matmul(b.T, Q), b)

    # e_min = np.matmul(np.matmul(b.T, Q), b)
    return b

def parse_Q(file="./H_0_5500.pkl"):

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
        if i == j:
            Q[i, j] = coe[key]
        else:
            Q[i, j] = Q[j, i] = coe[key] / 2

    return Q

def use_sample(Q = None, num_sweeps=1000, multiThread=False):
    if Q is None:
        binary_size = 40
        alpha = 0.01
        Q = alpha*symmetric(binary_size)

    b = np.zeros(len(Q))+1
    t0 = time.time()
    if multiThread:
        thread_num = 3
        b, e = DigitalAnnealing_multiThread(b, Q, num_sweeps=num_sweeps)
    else:
        b, e = DigitalAnnealing(b, Q, num_sweeps=num_sweeps, beta_range=(1, 50))

        # beta = np.geomspace(1, 50, num_sweeps)
        # b = DigitalAnnealing_numba(b, Q, num_sweeps=num_sweeps, beta=beta)
    t1 = time.time()

    print("time : {}".format(t1-t0))
    print("Minimum Energy : {}".format(np.min(e)))

    plt.figure()
    plt.plot(e)
    plt.show()

if __name__=="__main__":

    ############  time_step at 5500  ############
    # state = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 1,
    #               1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
    #               0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    #               1, 1, 1, 0, 1, 0, 0, 1, 0, 0])

    Q = parse_Q()
    # print(np.matmul(np.matmul(b.T, Q), b))
    use_sample(Q, num_sweeps=10000, multiThread=False)
