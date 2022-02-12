import json
import shutil
import glob
from coco import *
import os, cv2, shutil
from lxml import etree, objectify
from PIL import Image

def removeAndMkDir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    os.makedirs(path)


def save_annotations(filename, objs, filepath, dest_image_dir,dest_anno_dir):
    annopath = dest_anno_dir + "/" + filename[:-3] + "xml"  # 生成的xml文件保存路径
    dst_path = dest_image_dir + "/" + filename
    img_path = filepath
    img = cv2.imread(img_path)
    im = Image.open(img_path)
    if im.mode != "RGB":
        print(filename + " not a RGB image")
        im.close()
        return
    im.close()
    shutil.copy(img_path, dst_path)  # 把原始图像复制到目标文件夹
    E = objectify.ElementMaker(annotate=False)
    anno_tree = E.annotation(
        E.folder('1'),
        E.filename(filename),
        E.source(
            E.database('CKdemo'),
            E.annotation('VOC'),
            E.image('CK')
        ),
        E.size(
            E.width(img.shape[1]),
            E.height(img.shape[0]),
            E.depth(img.shape[2])
        ),
        E.segmented(0)
    )
    for obj in objs:
        E2 = objectify.ElementMaker(annotate=False)
        anno_tree2 = E2.object(
            E.name(obj[0]),
            E.pose(),
            E.truncated("0"),
            E.difficult(0),
            E.bndbox(
                E.xmin(obj[2]),
                E.ymin(obj[3]),
                E.xmax(obj[4]),
                E.ymax(obj[5])
            )
        )
        anno_tree.append(anno_tree2)
    etree.ElementTree(anno_tree).write(annopath, pretty_print=True)


def showbycv(coco, dataType, img, classes, origin_image_dir, dest_image_dir,dest_anno_dir,verbose=False):
    filename = img['file_name']
    filepath = os.path.join(origin_image_dir, dataType, filename)
    I = cv2.imread(filepath)
    annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
    anns = coco.loadAnns(annIds)
    objs = []
    for ann in anns:
        name = classes[ann['category_id']]
        if 'bbox' in ann:
            bbox = ann['bbox']
            xmin = (int)(bbox[0])
            ymin = (int)(bbox[1])
            xmax = (int)(bbox[2] + bbox[0])
            ymax = (int)(bbox[3] + bbox[1])
            obj = [name, 1.0, xmin, ymin, xmax, ymax]
            objs.append(obj)
            if verbose:
                cv2.rectangle(I, (xmin, ymin), (xmax, ymax), (255, 0, 0))
                cv2.putText(I, name, (xmin, ymin), 3, 1, (0, 0, 255))
    save_annotations(filename, objs, filepath,dest_image_dir,dest_anno_dir)
    if verbose:
        cv2.imshow("img", I)
        cv2.waitKey(0)

def catid2name(coco):  # 将名字和id号建立一个字典
    classes = dict()
    for cat in coco.dataset['categories']:
        classes[cat['id']] = cat['name']
        #print(str(cat['id'])+":"+cat['name'])
    return classes

def writeVOCClassesFile(classes, fCatFile):
    print("Going to write VOC classes to file ")
    newDict = dict()
    for eachOrigin in classes:
        #print("eachOrigin ", eachOrigin)
        newDict[classes[eachOrigin]] = int(eachOrigin)
    print("newDict ", newDict)
    json.dump(newDict, fCatFile)

def coco2voc(sourceDir, destDir, year='2017', verbose=False):
    dest_image_dir = os.path.join(destDir, 'JPEGImages')
    dest_anno_dir = os.path.join(destDir, 'Annotations')
    removeAndMkDir(dest_image_dir)
    removeAndMkDir(dest_anno_dir)
    origin_image_dir = sourceDir + '/JPEGImages'  # step 2 原始的coco的图像存放位置
    origin_anno_dir = sourceDir + '/annotations'  # step 3 原始的coco的标注存放位置

    dataTypes = ['train' + year, 'val' + year]
    destTrainValListDir = destDir + "/ImageSets/Main/"
    removeAndMkDir(destTrainValListDir)

    fCatFile = open(sourceDir + "/../" + "voc_classes.json", "w")
    for dataType in dataTypes:
        fileName = dataType[0:dataType.find(year)]
        fListFile = open(destTrainValListDir + fileName + ".txt", "w")
        annFile = 'instances_{}.json'.format(dataType)
        annpath = os.path.join(origin_anno_dir, annFile)
        coco = COCO(annpath)
        classes = catid2name(coco)
        if dataType.find('train') >= 0:
            writeVOCClassesFile(classes, fCatFile)
        imgIds = coco.getImgIds()
        # imgIds=imgIds[0:1000]#测试用，抽取10张图片，看下存储效果
        for imgId in imgIds:
            img = coco.loadImgs(imgId)[0]
            fListFile.write(str(img['file_name']).split(".")[0])
            fListFile.write('\n')
            showbycv(coco, dataType, img, classes, origin_image_dir, dest_image_dir,dest_anno_dir,verbose=False)
        fListFile.close()
    fCatFile.close()

if __name__ == "__main__":
    sourceDir = "E:/ubuntushare/data/coco2017"
    destDir = "E:/ubuntushare/data/coco2voc"
    coco2voc(sourceDir, destDir)



