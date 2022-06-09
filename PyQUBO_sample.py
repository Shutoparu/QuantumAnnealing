from pyqubo import Binary, Constraint, Placeholder, Array, OneHotEncInteger, LogEncInteger
import neal
from tqdm import tqdm
import numpy as np

#weights = [1, 3, 7, 9]
#values = [10, 2, 3, 6]
weights = np.random.random(2000)
values = np.random.random(2000)
max_weight = 10

# create the array of 0-1 binary variables
# representing the selection of the items n=len(values)
n = len(values)
items = Array.create('item', shape=n, vartype="BINARY")

item1, item2, item3, item4 = Binary("item1"), Binary("item2"), \
                             Binary("item3"), Binary("item4")

# define the sum of weights and values using variables
knapsack_weight = sum(
weights[i] * items[i] for i in range(n))

knapsack_value = sum(
values[i] * items[i] for i in range(n))

# define the coefficients of the penalty terms,
# lmd1 and lmd2, using Placeholder class
# so that we can change their values after compilation lmd1 = Placeholder("lmd1")

lmd1 = Placeholder("lmd1")
lmd2 = Placeholder("lmd2")

# create Hamiltonian and model

weight_one_hot = OneHotEncInteger("weight_one_hot",
     value_range=(1, max_weight), strength=lmd1)

a = LogEncInteger("w", value_range=(5, 20))

Ha = Constraint((weight_one_hot - knapsack_weight)**2,
     "weight_constraint")

Hb = knapsack_value

H = lmd2*Ha - Hb

model = H.compile()

sampler = neal.SimulatedAnnealingSampler()

feasible_sols = []

# search the best parameters: lmd1 and lmd2

# for lmd1_value in range(1, 10):
#     for lmd2_value in range(1, 10):
lmd1_value = 100
lmd2_value = 100
feed_dict = {'lmd1': lmd1_value, "lmd2":lmd2_value}

bqm = model.to_bqm(feed_dict=feed_dict)
bqm.normalize()
success = 0
for i in tqdm(range(1000)):
    sampleset = sampler.sample(bqm, num_reads=1,
                               num_sweeps=10000, beta_range=(1.0, 50.0), seed=i+1)

    dec_samples = model.decode_sampleset(sampleset, feed_dict=feed_dict)

    best = min(dec_samples, key=lambda x: x.energy)

    # store the feasible solution
    if not best.constraints(only_broken=True):
        feasible_sols.append(best)

    try:
        best_feasible = min(feasible_sols, key=lambda x: x.energy)
        # print(f"selection = {[best_feasible.sample[f'item[{i}]'] for i in range(n)]}")
        #
        # print(f"sum of the values = {-best_feasible.energy}")
        if -best_feasible.energy == 16:
            success+=1
        else:
            a = -best_feasible.energy
    except:
        continue

print('seccess rate : ' +str(success/1000))
