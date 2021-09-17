import torch as torch
import numpy as np

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape to vector
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # convert to one hot
    size.append(N)  #  reshape to original size
    return ones.view(*size)

inputX = torch.tensor([[1,2,3,1],
                       [2,1,2,3],
                       [2,1,3,2],
                       [1,2,2,3]])

print("inputX 1:", inputX.size())

inputX = inputX.unsqueeze(0)

print("inputX 2:", inputX.size())
print("inputX ", inputX)

gt = torch.LongTensor(inputX)

print("gt ", gt)

#gt_one_hot = get_one_hot(gt, 3)
#print(gt_one_hot)

gt = np.random.randint(0,5, size=[4, 15,15])
gt = torch.LongTensor(gt)

gt_one_hot = get_one_hot(gt, 5)
print(gt_one_hot.size())
gt_one_hot = gt_one_hot.permute(0, 3,  1, 2)
print(gt_one_hot.size())

