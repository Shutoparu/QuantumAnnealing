import numpy as np
import pickle
from pyqubo import Array, Binary, SubH

# a = Array.create("x", shape=(2, 3), vartype="BINARY")
# b1 = np.sum((a*a+5)*a)
# b2 = np.sum(a)
# c = SubH(b1, "h")
# model = c.compile()
# qubo, offset = model.to_qubo()
a = [1]
print(a*3*3)