import torch
import torch.utils.data as Data

BATCH_SIZE = 3
# demo data
x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1, 10)

torch_dataset = Data.TensorDataset(x,y)

print(type(torch_dataset))
print(type(x), type(y))

loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

for epoch in range(3):
    i = 0
    for batch_x, batch_y in loader:
        i += 1
        print('Epoch:{} | num:{} | batch_x:{} | batch_y:{}'
              .format(epoch, i, batch_x, batch_y))

