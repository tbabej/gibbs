from builders import ImageBuilder
from dwave import DWaveSampler

builder = ImageBuilder("/home/tbabej/pika.png")
model = builder.generate()
sampler = DWaveSampler()
embedding = builder.embedding_two()
import pprint
pprint.pprint(embedding)
print(embedding[125])
print(embedding[126])
results = sampler.sample(model, 10000, embedding)

import bpython
bpython.embed(locals_=locals())
