"""
Extract 2560 samples per class from the MNIS dataset, and save them on disk.
"""
import random
import pickle as pk

from akid import MNISTSource, ParallelSensor, AKID_DATA_PATH
from akid import backend as A
A.use_cuda(False)

# MNIST

# Clean data so it only has two class.
source = MNISTSource(work_dir=AKID_DATA_PATH + '/mnist', name='mnist')
s = ParallelSensor(
    source_in=source,
    # Do not shuffle training set for reproducible test
    sampler="sequence",
    name='mnist')
s.setup()

# Keep accumulating data until we have enough.
TARGET_CLASS = 0
SAMPLE_NUM = 2560
positive_samples = []
negative_samples = []
data = s.forward()
images = data[0]
labels = data[1]
count = 0
while count < 2560:
    for i, l in enumerate(labels):
        if l == TARGET_CLASS:
            positive_samples.append((images[i], l))
            count += 1
        elif len(negative_samples) < len(positive_samples):
            negative_samples.append((images[i], l))

assert len(positive_samples) == SAMPLE_NUM, len(positive_samples)
assert len(negative_samples) == SAMPLE_NUM, len(negative_samples)

# Save samples on disk.
samples = [i for i in positive_samples]
samples.extend(negative_samples)
random.shuffle(samples)
samples = [A.eval(i) for i in samples]

with open("mnist_binary.pk", 'wb') as f:
    pk.dump(samples, f)
