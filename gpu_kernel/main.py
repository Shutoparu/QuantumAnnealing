import numpy as np
import numpy.ctypeslib as ctplib
from ctypes import c_float, c_int, cdll, POINTER
import time


class DA:

    '''
    Attributes:
    qubo : np.ndarray
        the qubo matrix in 2D
    binary : np.ndarray
        the initial spin in 1D
    maxStep : int
        the maximum steps for the algorithm
    dim : int
        the dimention of the spin array
    time : float
        the time spent on the last execution of the algorithm. default value 0 is set
    '''

    def __init__(
        self,
        qubo: np.ndarray = np.array([[0, 1], [1, 0]]),
        binary: np.ndarray = None,
        maxStep: int = 10000,
    ) -> None:
        '''
        Parameters:
        qubo : np.ndarray
            the qubo matrix in 2D. elements will be parsed to np.float32 which is equivalent to "float" in C. default qubo matrix [[0,1],[1,0]] is used.
        binary : np.ndarray | None
            the initial spin in 1D with values between {-1,1}. elements will be parsed to np.float32 which is equivalent to "float" in C. if none then a random initial spin is generated
        maxStep : int
            the maximum steps for the algorithm. default value 10,000 is used
        '''

        self.qubo = qubo.astype(np.float32)
        self.maxStep = maxStep

        if np.shape(self.qubo)[0] != np.shape(self.qubo)[1]:
            print("qubo is not a square matrix")
            exit(-1)
        self.dim = np.shape(self.qubo)[0]

        if(binary is None):
            self.binary = np.zeros(self.dim)
            self.binary[-1] = 1
            self.binary = self.binary.astype(np.int32)
        else:
            self.binary = binary.astype(np.int32)

        if np.shape(self.qubo)[0] != np.shape(self.binary)[0]:
            print("qubo dimention and binary dimention mismatch")
            exit(-1)
        self.time = 0
        self.energy = 0

    def run(self) -> None:

        binary = ctplib.as_ctypes(self.binary)
        qubo = ctplib.as_ctypes(self.qubo.flatten())

        da = cdll.LoadLibrary("./lib/cudaDA.so")

        main = da.digitalAnnealingPy

        main.argtypes = [POINTER(c_int), POINTER(c_float), c_int, c_int]
        main.restype = c_float

        start = time.time()

        main(binary, qubo, self.dim, self.maxStep)

        end = time.time()

        self.time = end-start

        self.binary = ctplib.as_array(binary)


if __name__ == '__main__':

    np.random.seed(1)
    dim = 713
    maxStep = 7500
    qubo = 2 * np.random.rand(dim, dim).astype(np.float32) - 1
    qubo = (qubo + qubo.T) / 2
    binary = np.ones(dim).astype(np.float32)

    da = DA(qubo, binary, maxStep)
    da.run()
    print(da.time)
    print(da.energy)
