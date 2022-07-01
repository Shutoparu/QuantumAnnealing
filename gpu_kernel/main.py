import os
import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_float, c_int, cdll, POINTER
import time

np.random.seed(1)

dim = 1500
sweeps = 100000
qubo = 2 * np.random.rand(dim, dim).astype(np.float32) - 1
qubo = (qubo + qubo.T) / 2
qubo = qubo.flatten()
binary = np.ones(dim, dtype=np.int32)

binary = ctplib.as_ctypes(binary)
qubo = ctplib.as_ctypes(qubo)

os.system("nvcc --compiler-options -fPIC -shared -arch sm_70 " +
          "--maxrregcount=255 -o ./lib/cudaDA.so " +
          "cudaDigitalAnnealing.cu")
cudaDA = cdll.LoadLibrary("./lib/cudaDA.so")

main = cudaDA.digitalAnnealingPy

main.argtypes = [POINTER(c_int), POINTER(c_float), c_int, c_int]
main.restype = c_float

start = time.time()
energy = main(binary, qubo, dim, sweeps)
end = time.time()

binary = ctplib.as_array(binary)

print(energy)
print(binary)
print("spent time: ", end-start)
