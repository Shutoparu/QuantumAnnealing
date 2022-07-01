from mlb import pathloss_to_SINR, SINR_to_PRB, read_pkl
import cpu_kernel.digital_annealing as cpuda
from toQUBO import QUBO

#### read raw data ####
pathloss = read_pkl()

#### conver raw data to QUBO parameters ####
sinr, snr, rsrp = pathloss_to_SINR(pathloss)
prb, throughput = SINR_to_PRB(sinr, snr)

#### mlb configuration ####
ue_num, bs_num, time_step = sinr.shape
t = 5500
prb_t = prb[:, :, t]
throughput_t = throughput[:, :, t]
rsrp_t = rsrp[:, :, t]

#### toQUBO matrix ####
q = QUBO.paras_to_qubo(rsrp_t, throughput_t, prb_t, ue_num, bs_num)

#### cpu version of digital annealing algorithm ####
b, e = cpuda.annealing(q)
print(b)