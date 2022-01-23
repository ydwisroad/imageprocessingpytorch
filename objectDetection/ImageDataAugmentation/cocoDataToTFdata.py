# -*- encoding: utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import os
from multiprocessing import Process, Queue
import sys
import time
import random
import math
from pycocotools.coco import COCO

import tensorflow_datasets as tfds

annFile = 'E:/ubuntushare/data/coco2017/annotations/instances_train2017.json'
train_path = 'E:/ubuntushare/data/coco2017/train2017'
mytargetFolder = 'E:/ubuntushare/data/tfcoco2017/'

coco = COCO(annFile)
cores = 1
max_num = 1000

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
cats_dict = {}
for cat in cats:
    cats_dict[cat['id']] = cat['name']

# 获取COCO数据集中所有图像的ID
imgIds = coco.getImgIds()
print(len(imgIds))
# 构建训练集文件列表，里面的每个元素是路径名+图片文件名
train_images_filenames = os.listdir(train_path)
# 查找训练集的图片是否都有对应的ID，并保存到一个列表中
train_images = []
i = 1
total = len(train_images_filenames)
for image_file in train_images_filenames:
    if int(image_file[0:-4]) in imgIds:
        train_images.append(train_path + ',' + image_file)
    if i % 100 == 0 or i == total:
        print('processing image list %i of %i\r' % (i, total), end='')
    i += 1
random.shuffle(train_images)

all_cat = set()   # 保存目标检测所有的类别， COCO共定义了90个类别，其中只有80个类别有目标检测数据
imagefile_box = {}
# 获取每个图像的目标检测框的数据并保存
for item in train_images:
    boxes = [[], [], [], [], []]
    filename = item.split(',')[1]
    imgid = int(filename[0:-4])
    annIds = coco.getAnnIds(imgIds=imgid, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for ann in anns:
        bbox = ann['bbox']
        xmin = int(bbox[0])
        xmax = int(bbox[0] + bbox[2])
        ymin = int(bbox[1])
        ymax = int(bbox[1] + bbox[3])
        catid = ann['category_id']
        all_cat.add(cats_dict[catid])
        boxes[0].append(catid)
        boxes[1].append(xmin)
        boxes[2].append(ymin)
        boxes[3].append(xmax)
        boxes[4].append(ymax)
    imagefile_box[filename] = boxes

# 获取有目标检测数据的80个类别的名称
all_cat_list = list(all_cat)
all_cat_dict = {}
for i in range(len(all_cat_list)):
    all_cat_dict[all_cat_list[i]] = i
print(all_cat_dict)

# 把图像以及对应的检测框，类别等数据保存到TFRECORD


def make_example(image, height, width, label, bbox, filename):
    colorspace = b'RGB'
    channels = 3
    img_format = b'JPEG'
    return tf.train.Example(features=tf.train.Features(feature={
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
        'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
        'channels': tf.train.Feature(int64_list=tf.train.Int64List(value=[channels])),
        'colorspace': tf.train.Feature(bytes_list=tf.train.BytesList(value=[colorspace])),
        'img_format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_format])),
        'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
        'bbox_xmin': tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[0])),
        'bbox_xmax': tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[2])),
        'bbox_ymin': tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[1])),
        'bbox_ymax': tf.train.Feature(int64_list=tf.train.Int64List(value=bbox[3])),
        'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename]))
    }))

# 定义多进程函数用于生成TFRECORD文件


def gen_tfrecord(trainrecords, targetfolder, startnum, queue):
    tfrecords_file_num = startnum
    file_num = 0
    total_num = len(trainrecords)
    pid = os.getpid()
    queue.put((pid, file_num))
    writer = tf.io.TFRecordWriter(targetfolder + "train_" + str(tfrecords_file_num) + ".tfrecord")
    print("write to train ", targetfolder + "train_" + str(tfrecords_file_num) + ".tfrecord")
    for record in trainrecords:
        file_num += 1
        fields = record.split(',')
        img = cv2.imread(fields[0] + "/" + fields[1])
        height, width, _ = img.shape
        img_jpg = cv2.imencode('.jpg', img)[1].tobytes()
        bbox = imagefile_box[fields[1]]
        bbox[1] = [item for item in bbox[1]]  # xmin
        bbox[3] = [item for item in bbox[3]]  # xmax
        bbox[2] = [item for item in bbox[2]]  # ymin
        bbox[4] = [item for item in bbox[4]]  # ymax
        catnames = [cats_dict[item] for item in bbox[0]]
        label = [all_cat_dict[item] for item in catnames]
        ex = make_example(img_jpg, height, width, label, bbox[1:], fields[1].encode())
        writer.write(ex.SerializeToString())
        # 每写入100条记录，向父进程发送消息，报告进度
        if file_num % 100 == 0:
            queue.put((pid, file_num))
        if file_num % max_num == 0:
            writer.close()
            tfrecords_file_num += 1
            print("write to train ", targetfolder + "train_" + str(tfrecords_file_num) + ".tfrecord")
            writer = tf.io.TFRecordWriter(targetfolder + "train_" + str(tfrecords_file_num) + ".tfrecord")
    writer.close()
    queue.put((pid, file_num))

# 定义多进程处理


def process_in_queues(fileslist, cores, targetfolder):
    total_files_num = len(fileslist)
    each_process_files_num = int(total_files_num / cores)
    files_for_process_list = []
    for i in range(cores - 1):
        files_for_process_list.append(fileslist[i * each_process_files_num:(i + 1) * each_process_files_num])
    files_for_process_list.append(fileslist[(cores - 1) * each_process_files_num:])
    files_number_list = [len(l) for l in files_for_process_list]

    each_process_tffiles_num = math.ceil(each_process_files_num / max_num)

    queues_list = []
    processes_list = []
    for i in range(cores):
        queues_list.append(Queue())
        # queue = Queue()
        processes_list.append(Process(target=gen_tfrecord,
                                      args=(files_for_process_list[i], targetfolder,
                                            each_process_tffiles_num * i + 1, queues_list[i],)))

    for p in processes_list:
        Process.start(p)

    # 父进程循环查询队列的消息，并且每0.5秒更新一次
    while (True):
        try:
            total = 0
            progress_str = ''
            for i in range(cores):
                msg = queues_list[i].get()
                total += msg[1]
                progress_str += 'PID' + str(msg[0]) + ':' + str(msg[1]) + '/' + str(files_number_list[i]) + '|'
            progress_str += '\r'
            print(progress_str, end='')
            if total == total_files_num:
                for p in processes_list:
                    p.terminate()
                    p.join()
                break
            time.sleep(0.5)
        except:
            break
    return total


if __name__ == '__main__':
    print('Start processing train data using %i CPU cores:' % cores)
    starttime = time.time()
    #total_processed = process_in_queues(train_images, cores, targetfolder=mytargetFolder)
    endtime = time.time()
    #print('\nProcess finish, total process %i images in %i seconds' % (total_processed, int(endtime - starttime)),
    #      end='')

    val_dataset, dataset_info = tfds.load("coco",
                                          split="validation",
                                          with_info=True,
                                          data_dir=mytargetFolder,
                                          download=False)