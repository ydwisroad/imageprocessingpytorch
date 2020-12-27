import os
import json
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from toolbox.datasets import get_dataset
from toolbox.models import get_modelMoreParams
from toolbox.log import get_logger
import torchvision.transforms as transforms


def generatePredictSegmentation(modelName, modelPath, csvPath, inputPath, outputPath, n_classes=2):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_modelMoreParams(modelName, n_classes).to(device)
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()

    pd_label_color = pd.read_csv(csvPath, sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []

    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8')

    for eachFile in os.listdir(inputPath):
        img = Image.open(inputPath + '/' + eachFile)

        img = img.convert('RGB')

        transform_img = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
        )
        img = transform_img(img)
        #add one more dimension batch
        img = img.unsqueeze(0)
        img = img.to(device)
        #print("img " , img)

        out = model(img)
        out = F.log_softmax(out, dim=1)

        pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]
        pre1 = Image.fromarray(pre)
        pre1.save(outputPath + '/' + eachFile + '.png')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("-modelName", type=str, help="model name")
    parser.add_argument("-modelPath", type=str, help="model full path")

    parser.add_argument("-csvPath", type=str, help="csv path")
    parser.add_argument("-inputPath", type=str, help="input picture path")
    parser.add_argument("-outputPath", type=str, help="output pciture path")

    args = parser.parse_args()

    generatePredictSegmentation(args.modelName, args.modelPath, args.csvPath, args.inputPath, args.outputPath)

    #python generatePredictSegmentation.py - modelPath = F:\segmentRunResults\Final\yantaiDataset\enet - BCE\best_train_miou.pth -inputPath = F:\segmentRunResults\Final\yantaiDataset\test - outputPath = F:\segmentRunResults\Final\yantaiDataset\output - modelName = enet - csvPath = F:\segmentRunResults\Final\yantaiDataset\class_dict.csv

