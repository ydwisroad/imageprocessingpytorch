import torch
import torch.nn as nn
from torchvision import transforms, datasets
import json
import matplotlib.pyplot as plt
import random
import os
import shutil
import json
import time

import torch.optim as optim

from toolbox.log import get_logger
from toolbox.models import get_model
from toolbox.loss import get_loss


def run(cfg, logger):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # 所用数据集名称
    logger.info(f'Conf | use dataset {cfg["dataset"]}')
    # 所用数据增强方法
    logger.info(f'Conf | use augmentation {cfg["augmentation"]}')
    # 图片输入尺寸
    cfg['image_size'] = (cfg['image_h'], cfg['image_w'])
    logger.info(f'Conf | use image size {cfg["image_size"]}')

    data_transform = {
        "train": transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(cfg['image_h']),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(cfg['image_h']),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    #data_root = os.path.abspath(os.path.join(os.getcwd(), "../../data_test"))  # get data root path
    data_root = os.path.abspath(os.path.join(os.getcwd(), cfg["datasetpath"]))
    image_path = data_root + "/" + cfg["dataset"] + "/"  # car detection data set path

    train_dataset = datasets.ImageFolder(root=image_path+"train",
                                         transform=data_transform["train"])
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    #print("cla_dict ", cla_dict)

    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices_car.json', 'w') as json_file:
        json_file.write(json_str)

    # batch size大小
    logger.info(f'Conf | use batch_size {cfg["batch_size"]}')
    batch_size = cfg["batch_size"]
    num_workers = cfg['num_workers']
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers)

    validate_dataset = datasets.ImageFolder(root=image_path + "val",
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers)

    # 所用模型
    logger.info(f'Conf | use model {cfg["model_name"]}')
    net = get_model(cfg)
    #net = resnet34()

    num_classes = cfg["num_classes"]
    #net.fc = nn.Linear(inchannel, num_classes)
    net.to(device)

    #loss_function = nn.CrossEntropyLoss()
    logger.info(f'Conf | use loss function {cfg["loss"]}')
    loss_function = get_loss(cfg, weight=None).to(cfg['device'])

    #optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # 优化器 & 学习率衰减
    logger.info(f'Conf | use optimizer Adam, lr={cfg["lr"]}, weight_decay={cfg["weight_decay"]}')
    logger.info(f'Conf | use step_lr_scheduler every {cfg["lr_decay_steps"]} steps decay {cfg["lr_decay_gamma"]}')
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)

    best_acc = 0.0
    save_path = os.path.join(cfg['logdir'], 'best_train_' + cfg['model_name'] + '.pth') #'./resNet34.pth'

    # 训练 & 验证
    logger.info(f'Conf | use epoch {cfg["epoch"]}')

    for epoch in range(cfg['epoch']):
        # train
        net.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            images, labels = data
            #print("in image size ", images.size())
            #print("in labels size ", labels.size())
            optimizer.zero_grad()
            if (cfg['model_name'] == 'googlenet'):
                logits, aux_logits2, aux_logits1 = net(images.to(device))
                loss0 = loss_function(logits, labels.to(device))
                loss1 = loss_function(aux_logits1, labels.to(device))
                loss2 = loss_function(aux_logits2, labels.to(device))
                loss = loss0 + loss1 * 0.3 + loss2 * 0.3
            else:
                logits = net(images.to(device))
                loss = loss_function(logits, labels.to(device))

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # print train process
            rate = (step+1)/len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.4f}".format(int(rate*100), a, b, loss), end="")
        print()

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            for val_data in validate_loader:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))  # eval model only have last output layer
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += (predict_y == val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
            logger.info('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, val_accurate))
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / step, val_accurate))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/car_resnet.json",
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




