import os

img_path = "./VOCdevkit/VOC2012/JPEGImages/"
label_path = "./VOCdevkit/VOC2012/SegmentationClass/"

train_txt = "./VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"
val_txt = "./VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

train_path_txt = "./train.txt"
train_label_path_txt = "./train_labels.txt"

val_path_txt = "./val.txt"
val_label_path_txt = "./val_labels.txt"

name_file = open(train_txt, 'r')
path_file = open(train_path_txt, 'w')

for line in name_file.readlines():
    name = line.strip('\n')
    path = os.path.join(img_path, name)
    path_file.writelines(str(path) + '.jpg' + '\n')

name_file.close()
path_file.close()

