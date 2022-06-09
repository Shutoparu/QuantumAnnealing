import numpy as np
from pyqubo import Array, Placeholder, Constraint, SubH
import neal
import matplotlib.pyplot as plt
import re
import pickle
from tqdm import tqdm

# from pymatreader import read_mat

def sigmoid(arr, a=1, c=0):
    # input array dim: num_UE, num_BS

    return 1 / (1 + np.exp(-a * (arr - c)))

def pathloss_to_SINR(pathloss):
    power = 27
    power_W = 10**(power/10) * 10**(-3)
    k = 1.3803 * 10**(-23)
    T = 290
    BW = 100 * 10**6       # It's different from slide
    BBsr = 128.8 * 10**6   # Sampling rate

    # compute SNR/SINR
    num_UE, num_BS, num_MS = pathloss.shape # num_MS is number of measure time (time step number)
    
    SNR_dB = np.zeros((num_UE, num_BS, num_MS))
    SINR_dB = np.zeros((num_UE, num_BS, num_MS))
    RSRP_dBm = np.zeros((num_UE, num_BS, num_MS))
    No_dBm = 10 * np.log10(k*T*1000) + 10 * np.log10(BBsr) # Thermal noise
    No_W = 10**(No_dBm/10) * 10**(-3)

    RP = power_W * 10**(-1*pathloss/10)
    for i in range(num_BS):
        IF = np.sum(RP, axis=1) - RP[:,i,:]  # Interference (Watt)
        SINR_dB[:, i, :] = 10*np.log10(RP[:, i, :]/(No_W + IF))
        SNR_dB[:, i, :] = 10*np.log10(RP[:, i, :]/(No_W))
        RSRP_dBm[:, i, :] = np.around(10*np.log10(RP[:, i, :]*10**3))

    return SINR_dB, SNR_dB, RSRP_dBm

    
def SINR_to_PRB(SINR_dB, SNR_dB):

    # SINR dim : (num_UE, num_BS, len_time)
    UE_demand = 1.5 * 10**6           # bit/sec * measurement time
    num_UE, num_Bs, _ = SINR_dB.shape
    num_PRB = 273
    RE_per_PRB = 72
    slot_time = 0.5 * 10**(-3)
    Capacity = np.log2(1+10**(SINR_dB/10))
    # Require PRB
    R_PRB = np.ceil((UE_demand * slot_time)/(RE_per_PRB * Capacity))

    # throughput in one slot
    throughput = R_PRB * (RE_per_PRB * Capacity)

    return R_PRB/num_PRB, throughput

def read_pkl(file="7BS_150UE_pathloss.pkl"):
    with open(file, "rb") as f:
        data = pickle.load(f)

    # data = pkl['pathloss']
    return data

if __name__ == '__main__':

    pathloss = read_pkl()
    # data = read_mat('./pathloss.mat')
    # print("1",data['pathloss'].shape)
    # pathloss = data['pathloss']

    t_index = 5500
    UE_bound = 150


    SINR_dB, SNR_dB, RSRP_dBm = pathloss_to_SINR(pathloss)
    R_PRB, throughput = SINR_to_PRB(SINR_dB, SNR_dB)

    R_PRB = R_PRB[0:UE_bound,:,t_index]
    throughput = throughput[0:UE_bound,:,t_index]
    #RSRP = SNR_dB[0:UE_bound,:,t_index]
    RSRP = RSRP_dBm[0:UE_bound,:,t_index]

    num_UE = R_PRB.shape[0]
    num_BS = R_PRB.shape[1]
    print("num_UE, num_BS:", num_UE, num_BS)

    # Define Binary variables

    X = Array.create('X', shape=(num_UE, num_BS), vartype='BINARY')

    # Define lambda with Placeholder for tuning
    lmd0 = Placeholder("lmd0")
    lmd1 = Placeholder("lmd1")
    lmdc = Placeholder("lmdc")
    avg = np.sum(R_PRB * X)/num_BS
    c_avg = (np.max(RSRP)+np.min(RSRP))/2
    
    print("RSRP average:", c_avg)
    print("RSRP_minmal:", -np.sum(np.max(sigmoid(RSRP, c=c_avg), axis=1))/num_UE)


    H_0 = SubH(np.sum((np.sum(R_PRB * X, axis=0) - avg)**2)/num_BS, "H_0")
    H_1 = SubH(-1 * np.sum(sigmoid(RSRP, c=c_avg) * X)/num_UE, "H_1")
    H_c = Constraint(np.sum((np.sum(X, axis=1)-1)**2), label='one_hot')

    # H = lmd0 * H_0 + lmd1 *  H_1 + lmdc * H_c
    H = H_0

    model = H.compile()
    sampler = neal.SimulatedAnnealingSampler()

    # for lmdc_value in range(100, 101):
    #     for lmd0_value in range(10, 11):
    #         for lmd1_value in range(1, 2):

    # feed_dict = {'lmd0': lmd0_value, 'lmd1':
    #             lmd1_value, 'lmdc': lmdc_value}
    feed_dict = {'lmd0': 1, 'lmd1':
                1, 'lmdc': 1}
    qubo, offset = model.to_qubo(feed_dict=feed_dict)

    bqm = model.to_bqm(feed_dict=feed_dict)
    for i in tqdm(range(100)):
        sampleset = sampler.sample(bqm, num_reads=1,
                                num_sweeps=1000000, beta_range=[1, 50])
    decoded_samples = model.decode_sampleset(sampleset, feed_dict=feed_dict)
    best_sample = min(decoded_samples, key=lambda x: x.energy)

    print("Solution Energy:", best_sample.energy)
    # print("constraints:", best_sample.constraints()["one_hot"][0])
    # print("SubH", best_sample.subh)
    # print("H_0: ", np.sqrt(best_sample.subh['H_0']))
    # print("H_1: ", best_sample.subh['H_1'])
    solution = best_sample.sample
    print(solution)
