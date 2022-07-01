from mlb import pathloss_to_SINR, SINR_to_PRB, read_pkl
import cpu_kernel.digital_annealing as cpuda
from toQUBO import QUBO

import os
import math
import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_float, c_int, cdll, POINTER, pointer

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

#### reshape qubo matrix into 1D, and get its dimention ###
qubo = q.flatten().astype(np.float32)
dim = math.sqrt(qubo.shape()[0])
qubo = ctplib.as_ctypes(qubo)

#### create binary array ####
binary = np.ones(dim, dtype=np.int32)
binary = ctplib.as_ctypes(binary)

#### compile gpu algorithm library ####
os.system("nvcc --compiler-options -fPIC -shared -arch sm_70 " +
          "--maxrregcount=255 -o ./gpu_kernel/lib/cudaDA.so " +
          "./gpu_kernel/cudaDigitalAnnealing.cu")

#### load digital annealing gpu function ####
main = cdll.LoadLibrary("./gpu_kernel/lib/cudaDA.so").digitalAnnealingPy
main.argtypes = [POINTER(c_int), POINTER(c_float), c_int, c_int]
main.restype = c_float

#### gpu version of digital annealing algorithm ####
energy = main(binary, qubo , dim, num_sweeps=1000)

#### change binary array back to np.array type ####
binary = ctplib.as_array(binary)
