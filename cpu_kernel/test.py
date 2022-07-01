import numpy as np


class QUBO:
    @staticmethod
    def paras_to_qubo(rsrp, sinr, rb, bs_num, ue_num):
        assert (ue_num, bs_num) == rsrp.shape, "rsrp must have same shape with ({}, {})," \
                                               " but given {}".format(ue_num, bs_num, rsrp.shape)
        assert (ue_num, bs_num) == sinr.shape, "sinr must have same shape with ({}, {})," \
                                               " but given {}".format(ue_num, bs_num, sinr.shape)
        assert (ue_num, bs_num) == rb.shape, "rb must have same shape with ({}, {})," \
                                             " but given {}".format(ue_num, bs_num, rb.shape)

        bin_size = bs_num * ue_num
        Q = np.zeros([bin_size, bin_size])
        for j in range(bs_num):
            for i in range(ue_num):
                for k in range(ue_num):
                    Q[i+j*ue_num, k+j*ue_num] += rb[i, j]*rb[k, j]

        for k in range(bs_num):
            for l in range(ue_num):
                for j in range(bs_num):
                    for i in range(ue_num):
                        Q[i+j*ue_num, l+k*ue_num] += -1*rb[i, j]*rb[l, k]/bs_num

        Q /= bs_num
        return Q


if __name__=='__main__':
    bs_num = 5
    ue_num = 20
    rsrp = np.random.random([ue_num, bs_num])
    sinr = np.random.random([ue_num, bs_num])
    rb = np.random.random([ue_num, bs_num])
    matrix = QUBO.paras_to_qubo(rsrp, sinr, rb, bs_num, ue_num)