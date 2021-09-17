import random
import os
import shutil
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from toolbox.datasets import get_dataset
from toolbox.log import get_logger
from toolbox.models import get_model
from toolbox.loss import get_loss
from toolbox.loss import get_one_hot
from toolbox import evalution_segmentaion


def run(cfg, logger):
    # 所用数据集名称
    logger.info(f'Conf | use dataset {cfg["dataset"]}')
    # 所用数据增强方法
    logger.info(f'Conf | use augmentation {cfg["augmentation"]}')
    # 图片输入尺寸
    cfg['image_size'] = (cfg['image_h'], cfg['image_w'])
    logger.info(f'Conf | use image size {cfg["image_size"]}')

    # 获取训练集和验证集
    #trainset, valset, testset = get_dataset(cfg)
    trainset, valset = get_dataset(cfg)

    # batch size大小
    logger.info(f'Conf | use batch_size {cfg["batch_size"]}')

    train_loader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(valset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    # 所用模型
    logger.info(f'Conf | use model {cfg["model_name"]}')
    model = get_model(cfg)

    # 是否多gpu训练
    gpu_ids = [int(i) for i in list(cfg['gpu_ids'])]
    logger.info(f'Conf | use GPU {gpu_ids}')
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpu_ids)
    model = model.to(cfg["device"])
    print("Model ", model)

    # 优化器 & 学习率衰减
    logger.info(f'Conf | use optimizer Adam, lr={cfg["lr"]}, weight_decay={cfg["weight_decay"]}')
    logger.info(f'Conf | use step_lr_scheduler every {cfg["lr_decay_steps"]} steps decay {cfg["lr_decay_gamma"]}')
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size={cfg["lr_decay_steps"]}, gamma={cfg["lr_decay_gamma"]})

    # 损失函数 & 类别权重平衡
    logger.info(f'Conf | use loss function {cfg["loss"]}')
    #criterion = get_loss(cfg, weight=trainset.class_weight).to(cfg['device'])
    criterion = get_loss(cfg, weight=None).to(cfg['device'])

    # 训练 & 验证
    logger.info(f'Conf | use epoch {cfg["epoch"]}')

    for ep in range(cfg['epoch']):

        # training
        model.train()
        best = [0]
        train_loss = 0
        train_acc = 0
        train_miou = 0
        train_class_acc = 0
        train_p = 0
        train_r = 0
        train_f1 = 0

        for i, sample in enumerate(train_loader):
            # 载入数据
            #print("i " , i)
            img_data = sample['image'].to(cfg['device'])
            img_label = sample['label'].to(cfg['device'])

            img_labelNew = img_label
            if cfg['loss'] != 'crossentropyloss2D':
                #print("use more dimensional loss")
                img_labelNew = get_one_hot(sample['label'], cfg['n_classes'])
                img_labelNew = img_labelNew.permute(0, 3 , 1, 2).to(cfg['device'])

            #print("imge label size ", img_labelNew.size())
            # 训练
            out = model(img_data)
            #print("imge out pred size ", out.size())
            loss = criterion(out, img_labelNew)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 评估
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = img_label.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrix = evalution_segmentaion.eval_semantic_segmentation(pre_label, true_label, cfg)
            train_acc += eval_metrix['mean_class_accuracy']
            train_miou += eval_metrix['miou']
            train_class_acc += eval_metrix['class_accuracy']
            train_p += eval_metrix['p']
            train_r += eval_metrix['r']
            train_f1 += eval_metrix['f1']

        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epoch"]}] train loss={train_loss / len(train_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Train Acc={train_acc / len(train_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Train Mean IU={train_miou / len(train_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Train_class_acc={list(train_class_acc / len(train_loader))}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Train precision={train_p / len(train_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Train_recall={train_r / len(train_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Train_f1={train_f1 / len(train_loader):.5f}')

        if max(best) <= train_miou / len(train_loader):
            best.append(train_miou / len(train_loader))
            torch.save(model.state_dict(), os.path.join(cfg['logdir'], 'best_train_miou.pth'))

        #scheduler.step()
        net = model.eval()
        eval_loss = 0
        eval_acc = 0
        eval_miou = 0
        eval_class_acc = 0
        eval_p = 0
        eval_r = 0
        eval_f1 = 0

        for j, sample in enumerate(val_loader):
            valImg = sample['image'].to(cfg['device'])
            valLabel = sample['label'].to(cfg['device'])

            valLabelNew = valLabel
            if cfg['loss'] != 'crossentropyloss2D':
                #print("use more dimensional loss")
                valLabelNew = get_one_hot(sample['label'], cfg['n_classes'])
                valLabelNew = valLabelNew.permute(0, 3 , 1, 2).to(cfg['device'])

            out = net(valImg)
            loss = criterion(out, valLabelNew)
            eval_loss = loss.item() + eval_loss
            pre_label = out.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = valLabel.data.cpu().numpy()
            true_label = [i for i in true_label]

            eval_metrics = evalution_segmentaion.eval_semantic_segmentation(pre_label, true_label, cfg)
            eval_acc = eval_metrics['mean_class_accuracy'] + eval_acc
            eval_miou = eval_metrics['miou'] + eval_miou
            eval_class_acc = eval_metrix['class_accuracy'] + eval_class_acc
            eval_p += eval_metrix['p']
            eval_r += eval_metrix['r']
            eval_f1 += eval_metrix['f1']

        logger.info(f'Iter | [{ep + 1:3d}/{cfg["epoch"]}] valid loss={eval_loss / len(val_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Valid Acc={eval_acc / len(val_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Valid Mean IU={eval_miou / len(val_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Valid Class Acc={list(eval_class_acc / len(val_loader))}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Valid precision={eval_p / len(val_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Valid recall={eval_r / len(val_loader):.5f}')
        logger.info(f'Test | [{ep + 1:3d}/{cfg["epoch"]}] Valid f1={eval_f1 / len(val_loader):.5f}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
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

    # 训练的各种记录的保存目录
    logdir = f'run/{cfg["dataset"]}/{time.strftime("%Y-%m-%d-%H-%M")}-{random.randint(1000, 10000)}'  # the same time as log + random id
    os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)

    logger.info(f'Conf | use logdir {logdir}')

    cfg['device'] = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cfg['logdir'] = logdir

    run(cfg, logger)