from builders import ImageBuilder
from dwave import DWaveSampler

model = ImageBuilder("/home/tbabej/pika.png").generate()
sampler = DWaveSampler()

results = sampler.sample(model, 10000)

import bpython
bpython.embed(locals_=locals())
