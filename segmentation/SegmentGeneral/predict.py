import os
import json
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from toolbox.datasets import get_dataset
from toolbox.models import get_model
from toolbox.log import get_logger
from toolbox import evalution_segmentaion


def predict(cfg, runid, use_pth='best_train_miou.pth'):

    dataset = cfg['dataset']
    train_logdir = f'run/{dataset}/{runid}'

    test_logdir = os.path.join('./results', dataset, runid)
    logger = get_logger(test_logdir)

    logger.info(f'Conf | use logdir {train_logdir}')
    logger.info(f'Conf | use dataset {cfg["dataset"]}')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 测试集
    trainset, valset = get_dataset(cfg)
    #temporarily use valset as testset
    testset = valset

    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])

    # model
    model = get_model(cfg).to(device)
    modelPath = os.path.join(train_logdir, use_pth)
    print("model Path ", modelPath)
    model.load_state_dict(torch.load(modelPath))

    pd_label_color = pd.read_csv(trainset.file_path[2], sep=',')
    name_value = pd_label_color['name'].values
    num_class = len(name_value)
    colormap = []
    for i in range(num_class):
        tmp = pd_label_color.iloc[i]
        color = [tmp['r'], tmp['g'], tmp['b']]
        colormap.append(color)

    cm = np.array(colormap).astype('uint8')

    test_acc = 0
    test_miou = 0
    test_class_acc = 0
    test_mpa = 0

    for i, sample in enumerate(test_loader):
        valImg = sample['image'].to(device)
        valLabel = sample['label'].long().to(device)
        print("valImg", valImg)
        out = model(valImg)
        out = F.log_softmax(out, dim=1)

        pre_label = out.max(1)[1].squeeze().cpu().data.numpy()
        pre = cm[pre_label]
        pre1 = Image.fromarray(pre)
        pre1.save(test_logdir + '/' + str(i) + '.png')

        pre_label = out.max(dim=1)[1].data.cpu().numpy()
        pre_label = [i for i in pre_label]

        true_label = valLabel.data.cpu().numpy()
        true_label = [i for i in true_label]

        eval_metrix = evalution_segmentaion.eval_semantic_segmentation(pre_label, true_label, cfg)
        test_acc = eval_metrix['mean_class_accuracy'] + test_acc
        test_miou = eval_metrix['miou'] + test_miou
        test_mpa = eval_metrix['pixel_accuracy'] + test_mpa

        if len(eval_metrix['class_accuracy']) < 12:
            eval_metrix['class_accuracy'] = 0
            test_class_acc = test_class_acc + eval_metrix['class_accuracy']
        else:
            test_class_acc = test_class_acc + eval_metrix['class_accuracy']

    logger.info(f'Test | Test Acc={test_acc / (len(test_loader)):.5f}')
    logger.info(f'Test | Test Mpa={test_mpa / (len(test_loader)):.5f}')
    logger.info(f'Test | Test Mean IU={test_miou / (len(test_loader)):.5f}')
    #logger.info(f'Test | Test_class_acc={list(test_class_acc / (len(test_loader)))}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="predict")
    parser.add_argument("-id", type=str, help="predict id")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/camvid_linknet.json",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    #args.id = '2020-11-02-15-14-4953'

    predict(cfg, args.id)
