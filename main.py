from pyqubo import Binary, Placeholder, Constraint
import neal

x, y = Binary('x'), Binary('y')
lmd = Placeholder('lmd')

H = x*x + 4.0*x*y + y*y + Constraint(lmd*(1-x), label='one_hot')
model = H.compile()
qubo, offset = model.to_bqm()

feed_dic = {'lmd':1}
sampler = neal.SimulatedAnnealingSampler(feed_dict=feed_dic)
sampleset = sampler.sample(qubo)

dec_samples = model.decode_sampleset(sampleset)
