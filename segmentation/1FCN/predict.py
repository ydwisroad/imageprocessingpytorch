import pandas as pd
import numpy as np
import torch as t
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from dataset import LoadDataset
from Models import FCN
import cfg

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg.DATASET[1]

Load_test = LoadDataset([cfg.TEST_ROOT, cfg.TEST_LABEL], cfg.crop_size)
test_data = DataLoader(Load_test, batch_size=1, shuffle=True, num_workers=4)

net = FCN.FCN(num_class).to(device)
net.load_state_dict(t.load(cfg.RESULT_ROOT + "/weights/xxx.pth"))
net.eval()

pd_label_color = pd.read_csv(cfg.class_dict_path, sep=',')
name_value = pd_label_color['name'].values
num_class = len(name_value)
colormap = []
for i in range(num_class):
	tmp = pd_label_color.iloc[i]
	color = [tmp['r'], tmp['g'], tmp['b']]
	colormap.append(color)

cm = np.array(colormap).astype('uint8')

dir = cfg.RESULT_ROOT + "/result_pics/"

for i, sample in enumerate(test_data):
	valImg = sample['img'].to(device)
	valLabel = sample['label'].long().to(device)
	out = net(valImg)
	out = F.log_softmax(out, dim=1)
	pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
	pre = cm[pre_label]
	pre1 = Image.fromarray(pre)
	pre1.save(dir + str(i) + '.png')
	print('Done')