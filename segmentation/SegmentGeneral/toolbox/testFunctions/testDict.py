import numpy as np
from array import array
import torch as torch

labels = [0, 1, 1, 0, 1, 0, 1]
arrLabels = np.array(labels)
print(torch.from_numpy(arrLabels))

label2idx = dict()
for i in np.array(labels):
    if not (i in label2idx):
        label2idx[i] = i

print(label2idx)