import torch.nn as nn
import torch.nn.functional as F

if __name__ == '__main__':
    import torch

    predict = torch.randn((2, 21, 512, 512))
    gt = torch.randint(1, 20, (2, 512, 512))

    loss_function = nn.CrossEntropyLoss()
    result = loss_function(predict, gt)
    print(result)
