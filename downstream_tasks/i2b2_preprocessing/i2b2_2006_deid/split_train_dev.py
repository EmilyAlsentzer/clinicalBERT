with open('train_dev.conll', 'r') as f:
    sents = f.read().strip().split('\n\n')

print(len(sents))
import random

random.seed(555)

n = len(sents)
ind = int(0.7*n)

train_sents = sents[:ind]
dev_sents = sents[ind:]

print(len(train_sents))
with open('train.conll', 'w') as f:
    f.write('\n\n'.join(train_sents))

print(len(dev_sents))
with open('dev.conll', 'w') as f:
    f.write('\n\n'.join(dev_sents))
