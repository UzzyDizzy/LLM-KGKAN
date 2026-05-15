# sampling.py
import random

def make_batches(data, batch_size):
    idx=list(range(len(data)))
    random.shuffle(idx)

    for i in range(0,len(idx),batch_size):
        yield [data[j] for j in idx[i:i+batch_size]]
