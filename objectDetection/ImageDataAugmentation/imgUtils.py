import numpy as np
import cv2
import os

import random
from os.path import join

from augmentImages import *
from Helpers import *
from util import *

#https://blog.csdn.net/qq_20622615/article/details/80929746
def letterbox(img: np.ndarray,
              new_shape=(416, 416),
              color=(114, 114, 114),
              auto=True,
              scale_fill=False,
              scale_up=True):
    """
    将图片缩放调整到指定大小
    :param img: 输入的图像numpy格式
    :param new_shape: 输入网络的shape
    :param color: padding用什么颜色填充
    :param auto:
    :param scale_fill: 简单粗暴缩放到指定大小
    :param scale_up:  只缩小，不放大
    :return:
    """

    shape = img.shape[:2]  # [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scale_up:  # only scale down, do not scale up (for better test mAP) 对于大于指定输入大小的图片进行缩放,小于的不变
        r = min(r, 1.0)

    # compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimun rectangle 保证原图比例不变，将图像最大边缩放到指定大小
        # 这里的取余操作可以保证padding后的图片是32的整数倍(416x416)，如果是(512x512)可以保证是64的整数倍
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scale_fill:  # stretch 简单粗暴的将图片缩放到指定尺寸
        dw, dh = 0, 0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # wh ratios

    dw /= 2  # divide padding into 2 sides 将padding分到上下，左右两侧
    dh /= 2

    # shape:[h, w]  new_unpad:[w, h]
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))  # 计算上下两侧的padding
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))  # 计算左右两侧的padding

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def centerxywhYoloTox1y1x2y2(width, height, xCenter, yCenter, widthPer, heightPer):
    x = xCenter * width
    y = yCenter * height

    return x - (width * widthPer)/2, y - (height * heightPer)/2, x + (width * widthPer) /2, y + (height * heightPer)/2


def x1y1x2y2TocenterxywhYolo(width, height, xmin, ymin, xmax, ymax):
    xCenter = (xmax + xmin) / (2 * width)
    yCenter = (ymin + ymax) / (2 * height)
    widthPer = (xmax - xmin) / width
    heightPer = (ymax - ymin) / height

    return xCenter, yCenter, widthPer, heightPer


def addLetterBoxesForFolder(imgPath, labelPath, saveImgPath, saveLabelPath, newShape=[512, 512], color=(0,0,0)):
    for eachImageFile in os.listdir(imgPath):
        fileName = eachImageFile.split(".")[0]
        eachLabelFile = os.path.join(labelPath + "/" + fileName + ".txt")

        fullImgPath = imgPath + "/" + eachImageFile
        print("fullImgPath ", fullImgPath)
        img_o = cv2.imread(fullImgPath)  # BGR
        img, ratio, pad = letterbox(img_o, newShape, auto=True, color=(112,112,112), scale_fill=False)

        originSize = img_o.shape
        labelsYoloPos = [f.strip() for f in open(eachLabelFile).readlines()]
        boxes_list = []
        for eachLabel in labelsYoloPos:
            print(" originSize ", originSize[1], " ", originSize[0], " labelsYoloPos ", eachLabel)
            eachLabelSplit = eachLabel.split(" ")
            x1,y1, x2, y2 = centerxywhYoloTox1y1x2y2(float(originSize[1]),  float(originSize[0]),
                                                     float(eachLabelSplit[1]), float(eachLabelSplit[2]),
                                                     float(eachLabelSplit[3]), float(eachLabelSplit[4]))
            boxes_list.append([x1, y1, x2, y2])

        oneLabel = eachLabelSplit[0]
        boxes_list =np.array(boxes_list)
        new_boxes = np.zeros_like(boxes_list)
        print(" ratio ", ratio, " pad", pad)
        new_boxes[:, 0] = ratio[0] * boxes_list[:, 0] + pad[0]  # pad width
        new_boxes[:, 1] = ratio[1] * boxes_list[:, 1] + pad[1]  # pad height
        new_boxes[:, 2] = ratio[0] * boxes_list[:, 2] + pad[0]
        new_boxes[:, 3] = ratio[1] * boxes_list[:, 3] + pad[1]
        print("new_boxes ", new_boxes)

        print("img shape ", img.shape)
        xCenter, yCenter, widthPer, heightPer = x1y1x2y2TocenterxywhYolo(img.shape[1], img.shape[0],
                                                                         new_boxes[0][0], new_boxes[0][1],new_boxes[0][2], new_boxes[0][3])
        #write new text file with object pos
        saveLabelFile = os.path.join(saveLabelPath + "/" + fileName + ".txt")
        label_file = open(saveLabelFile, 'w')
        label_file.write('%s ' % (oneLabel))
        label_file.write('%.6f ' % (xCenter))
        label_file.write('%.6f ' % (yCenter))
        label_file.write('%.6f ' % (widthPer))
        label_file.write('%.6f ' % (heightPer))
        label_file.write('\n')
        label_file.close()

        fullSaveImgPath = saveImgPath + "/" + eachImageFile
        print("fullSaveImgPath ", fullSaveImgPath)
        new_im = cv2.imwrite(fullSaveImgPath, img)

#crop objects which are annotated in image, save as an independent image.
def cropObjectsFromImage(imgFile, labelFile, saveImagePath, countParent):
    image = cv2.imread(imgFile)

    labelsYoloPos = [f.strip() for f in open(labelFile).readlines()]
    boxes_list = []
    count = 0
    for eachLabel in labelsYoloPos:
        eachLabelSplit = eachLabel.split(" ")
        x1, y1, x2, y2 = centerxywhYoloTox1y1x2y2(float(image.shape[1]), float(image.shape[0]),
                                                  float(eachLabelSplit[1]), float(eachLabelSplit[2]),
                                                  float(eachLabelSplit[3]), float(eachLabelSplit[4]))
        region = image[int(y1):int(y2), int(x1):int(x2)]

        object_im = cv2.imwrite(saveImagePath + "/" + str(eachLabelSplit[0]) + "_" + str(countParent) + "_"
        + str(count) + ".png", region)
        count = count + 1

def cropObjectsFromImagePath(imgFilePath, labelFilePath, saveImagePath):
    print("This is the start of cropObjectsFromImagePath")
    count = 0
    for eachImageFile in os.listdir(imgFilePath):
        fileName = eachImageFile.split(".")[0]
        eachLabelFile = os.path.join(labelFilePath + "/" + fileName + ".txt")
        print("eachLabelFile ", eachLabelFile)
        fullImgPath = imgFilePath + "/" + eachImageFile
        print("fullImgPath ", fullImgPath)
        cropObjectsFromImage(fullImgPath, eachLabelFile, saveImagePath, count)
        count = count + 1

def copySmallObjectsToOneBlankImage(oneEmptyImage, smallObjectsFolder, saveFolder, times, imagesOutputCount):
    small_imgs_dir = []
    for eachImageFile in os.listdir(smallObjectsFolder):
        small_imgs_dir.append(smallObjectsFolder  + "/" + eachImageFile)
    random.shuffle(small_imgs_dir)

    #image = cv2.imread(oneEmptyImage)
    for iCount in range(imagesOutputCount):
        #randomly select small objects as paste candidate
        small_imgs = np.random.choice(small_imgs_dir, times)
        copysmallobjects2(oneEmptyImage, None, saveFolder, small_imgs, times, iCount, area_max=40000, area_min=200)


def generateAugmentedObjects(smallObjectsFolder, outputFolder):
    iCount = 0
    for eachSmallImageFile in os.listdir(smallObjectsFolder):
        print("process eachSmallImageFile ", eachSmallImageFile)
        img = cv2.imread(smallObjectsFolder + "/" + eachSmallImageFile)

        smallObjName = os.path.basename(eachSmallImageFile)
        objName = smallObjName.split(".")[0]

        remain = iCount % 5
        outPath = outputFolder + "/" + objName + str(remain) + ".png"
        imgOut = None
        if (remain==0):
            imgOut = horizontalFlip(img)
        elif (remain==1):
            imgOut = randomHSV(img)
        elif (remain ==2):
            imgOut = gauss_noise(img)
        elif (remain == 3):
            imgOut = sp_noise(img)
        elif (remain == 4):
            imgOut = sharpenImage(img)

        cv2.imwrite(outPath,imgOut)

        iCount = iCount + 1




###############################
##############################
# a series of opencv transformer operations added below
def horizontalFlip(img):
    img = img[:, ::-1, :]
    return img

def shear(img, shear_factor):
    shear_factor = random.uniform(shear_factor, 1)
    w, h = img.shape[1], img.shape[0]

    if shear_factor < 0:
        img = horizontalFlip(img)
    M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])

    nW = img.shape[1] + abs(shear_factor * img.shape[0])

    img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

    if shear_factor < 0:
        img = horizontalFlip(img)

    img = cv2.resize(img, (w, h))

    return img


def randomHSV(img, hue=1, saturation=1, brightness=1):
        #if type(hue) != tuple:
        #    hue = (-hue, hue)

        #if type(saturation) != tuple:
        #    saturation = (-saturation, saturation)

        #if type(brightness) != tuple:
        #    brightness = (-brightness, brightness)

        hue = random.randint(-hue, hue)
        saturation = random.randint(-saturation, saturation)
        brightness = random.randint(-brightness,brightness)

        img = img.astype(int)

        a = np.array([hue, saturation, brightness]).astype(int)
        img += np.reshape(a, (1, 1, 3))

        img = np.clip(img, 0, 255)
        img[:, :, 0] = np.clip(img[:, :, 0], 0, 179)

        img = img.astype(np.uint8)

        return img

def gauss_noise(img,sigma=25):
	temp_img = np.float64(np.copy(img))
	h = temp_img.shape[0]
	w = temp_img.shape[1]
	noise = np.random.randn(h,w) * sigma
	noisy_img = np.zeros(temp_img.shape, np.float64)
	if len(temp_img.shape) == 2:
		noisy_img = temp_img + noise
	else:
		noisy_img[:,:,0] = temp_img[:,:,0] + noise
		noisy_img[:,:,1] = temp_img[:,:,1] + noise
		noisy_img[:,:,2] = temp_img[:,:,2] + noise
	# noisy_img = noisy_img.astype(np.uint8)
	return noisy_img

def sp_noise(image,prob = 0.03):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def modifybrightness(img):
    # convert img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # do adaptive threshold on gray image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 35)

    # make background of input white where thresh is white
    result = img.copy()
    result[thresh == 255] = (255, 255, 255)

    return result

def sharpenImage(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    modified_image = cv2.filter2D(image, -1, kernel)

    return modified_image

def draw_rectForObj(im, cords, color=None):
    """Draw the rectangle on the image
    Parameters
    ----------
    im : numpy.ndarray
        numpy image

    cords: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    Returns
    -------
    numpy.ndarray
        numpy image with bounding boxes drawn on it

    """
    im = im.copy()

    cords = cords[:, :5]
    cords = cords.reshape(-1, 5)
    if not color:
        color = [255, 255, 255]
    #print("got cords ", cords)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cord in cords:
        pt1, pt2 = (cord[0], cord[1]), (cord[2], cord[3])
        indexNum = cord[4]
        pt1 = int(float(pt1[0])), int(float(pt1[1]))
        pt2 = int(float(pt2[0])), int(float(pt2[1]))
        #print("pt1 pt2 ", pt1, " ", pt2)
        im = cv2.putText(im.copy(), str(indexNum), pt1,font, 1.2, (255, 255, 255), 2)
        im = cv2.rectangle(im.copy(), pt1, pt2, color)
    return im

def addObjRectToImages(rootPath, outputImagesPath):
    print("start to addObjRectToImages")
    allImgFiles = os.listdir(rootPath + "/images/")
    allImgFiles.sort()

    for eachImgFile in allImgFiles:
        if (not eachImgFile.endswith("jpg") and not eachImgFile.endswith("png")):
            continue
        fullImgPath = rootPath + '/images/' + eachImgFile
        fileParts = eachImgFile.split(".")
        labelFilePath = rootPath +  '/labels/' + fileParts[0] + '.txt'
        print(" each image path ", fullImgPath)
        img = cv2.imread(fullImgPath)
        iw, ih, _ = img.shape

        #...To be added
        bboxes = None
        with open(labelFilePath, "r") as labelFile:
            for line in labelFile:
                parts = line.split(" ")
                x1, y1,x2,y2 = centerxywhYoloTox1y1x2y2(iw, ih, float(parts[1]),
                                                        float(parts[2]), float(parts[3]),float(parts[4]))
                eachPos = np.array([[x1, y1, x2, y2, parts[0]]])
                if (bboxes is None):
                    bboxes = eachPos
                else:
                    bboxes = np.concatenate((bboxes, eachPos), axis=0)
            print(" bboxes ", bboxes)
        imgRects = draw_rectForObj(img, bboxes)

        outputRectImagePath = outputImagesPath + "/" + eachImgFile
        cv2.imwrite(outputRectImagePath, imgRects)



























