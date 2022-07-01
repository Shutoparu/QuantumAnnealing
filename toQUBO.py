import numpy as np
from pyqubo import SubH, Array

class equation:
    def __init__(self, key):
        self.key = key
        # self.value = value

    def __mul__(self, other):
        if isinstance(other, equation):
            self.key = "({}*{})".format(self.key, other.key)
        else:
            self.key = "({}*{})".format(self.key, other)
        return self

    def __sub__(self, other):
        if isinstance(other, equation):
            self.key = "({}-{})".format(self.key, other.key)
        else:
            self.key = "({}-{})".format(self.key, other)
        return self

    def __add__(self, other):
        if isinstance(other, equation):
            self.key = "({}+{})".format(self.key, other.key)
        else:
            self.key = "({}+{})".format(self.key, other)
        return self

    def __truediv__(self, other):
        assert not isinstance(other, equation), "can't div by binary variable"
        self.key = "({}/{})".format(self.key, other)
        return self


class QUBO:
    @staticmethod
    def paras_to_qubo(rsrp, sinr, rb, ue_num, bs_num):
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

    def __init__(self):
        self.bin_vars = {}

    # @staticmethod
    def generator(self, var_name):
        '''
        :param var_name: binary variable name, has string type
        :return: build variable dict(bin_vars) and return equation object to describe formula
        '''
        self.bin_vars[var_name] = 0
        return equation(var_name)

    # @staticmethod
    def create_array(self, shape, var_name):
        '''
        :param shape: binary variable array shape, list type
        :param var_name: variable name, string type
        :return: return a given shape equation obj list which is initialized by given variable name
        '''
        if len(shape) > 1:
            length = shape[0]
            return [self.create_array(shape[1:], "{}[{}]".format(var_name, i)) for i in range(length)]
        else:
            length = shape[0]
            return [self.generator("{}[{}]".format(var_name, i)) for i in range(length)]

    # @staticmethod
    def multiply(self, array, coe):
        if isinstance(array[0], list):
            op = False
        else:
            op = True

        length = array.__len__()

        if not isinstance(coe, list) and not isinstance(coe, np.ndarray):
            coe = np.repeat(coe, length)

        if op:
            return [(array[i] * coe[i]) for i in range(length)]
        else:
            return [self.multiply(array[i], coe[i]) for i in range(length)]

    def add(self, array, coe):
        if isinstance(array[0], list):
            op = False
        else:
            op = True

        length = array.__len__()

        if not isinstance(coe, list) and not isinstance(coe, np.ndarray):
            coe = np.repeat(coe, length)

        if op:
            return [(array[i] + coe[i]) for i in range(length)]
        else:
            return [self.add(array[i], coe[i]) for i in range(length)]

    def sub(self, array, coe):
        if isinstance(array[0], list):
            op = False
        else:
            op = True

        length = array.__len__()

        if not isinstance(coe, list) and not isinstance(coe, np.ndarray):
            coe = np.repeat(coe, length)
        elif coe.__len__() == 1:
            coe = coe * length

        if op:
            return [(array[i] - coe[i]) for i in range(length)]
        else:
            return [self.sub(array[i], coe[i]) for i in range(length)]

    def div(self, array, coe):
        if isinstance(array[0], list):
            op = False
        else:
            op = True

        length = array.__len__()

        if not isinstance(coe, list) and not isinstance(coe, np.ndarray):
            coe = np.repeat(coe, length)

        if op:
            return [(array[i] / coe[i]) for i in range(length)]
        else:
            return [self.div(array[i], coe[i]) for i in range(length)]

    def square(self, array):
        if isinstance(array[0], list):
            op = False
        else:
            op = True

        length = array.__len__()

        if op:
            return [(array[i] * array[i]) for i in range(length)]
        else:
            return [self.multiply(array[i], array[i]) for i in range(length)]

    def join(self, array):
        length = array.__len__()
        tmp = array[0].key
        for i in range(1, length):
            tmp = "({}+{})".format(tmp, array[i].key)
        return equation(tmp)

    def sum(self, array, axis=None):
        length = array.__len__()
        if axis is None:
            if isinstance(array[0], list):
                equ_tmp = [self.sum(array[i]) for i in range(length)]
                return [self.join(equ_tmp)]
            else:
                return self.join(array)
        else:
            if axis == 0:
                tmp = array[0]
                for i in range(1, length):
                    tmp = self.add(tmp, array[i])
                return [tmp]
            else:
                return [self.sum(array[i], axis - 1) for i in range(length)]

    def toQUBO(self, array):
        pass


if __name__=="__main__":
    Q = QUBO()
    size = [2, 5]

    # rsrp = np.random.random(size)
    # prb = np.random.random(size)
    # x = Q.create_array(size, "x")
    # avg = Q.div(Q.sum(Q.multiply(x, prb)), 5)
    # # h = Q.div(Q.sum(Q.square(Q.sum(Q.multiply(x, rsrp), axis=0))), 20)
    # h = Q.div(Q.sum(Q.square(Q.sub(Q.sum(Q.multiply(x, rsrp), axis=0), avg))), 20)
    # print(avg[0].key)

    bs_num = 5
    ue_num = 20
    rsrp = np.random.random([ue_num, bs_num])
    sinr = np.random.random([ue_num, bs_num])
    rb = np.random.random([ue_num, bs_num])
    matrix = Q.paras_to_qubo(rsrp, sinr, rb, bs_num, ue_num)

    X = Array.create('X', shape=(ue_num, bs_num), vartype='BINARY')
    avg = np.sum(rb * X)/bs_num
    H_0 = SubH(np.sum((np.sum(rb * X, axis=0) - avg) ** 2) / bs_num, "H_0")
    H = H_0
    model = H.compile()
    qubo, offset = model.to_qubo()
    print(qubo['X[0][0]', 'X[0][0]'])
    print(qubo['X[10][0]', 'X[10][0]'])
    print("Pause")