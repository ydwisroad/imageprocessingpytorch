import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 4邻域的连通域和 8邻域的连通域
# [row, col]
NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]

NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]

#第二遍扫描
def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    return binary_img, points

#四领域或八领域判断
def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row+offset[0]), rows-1)
                neighbor_col = min(max(0, col+offset[1]), cols-1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    print('第一遍扫描：',binary_img.shape)
    print('开始第二遍...')
    return binary_img

# binary_img: bg-0, object-255; int
#第一遍扫描
def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError
    binary_img = neighbor_value(binary_img, offsets, False)

    return binary_img

def connectArea(img, outputPath, img_name):
    ret, binary = cv2.threshold(img, 75, 255, cv2.THRESH_BINARY)  # 灰度转二值图像
    kernel = np.ones((21, 21), np.uint8)  # 给图像闭运算定义核
    kernel_1 = np.ones((101, 101), np.uint8)  # 给图像开运算定义核
    # 图像先闭运算再开运算可以过滤孤立的物体， 将密集物体区域形成一片连通区
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_1)
    # 给图像的边缘像素设定为255，否则下面连通区的检测无法识别贴在图像边缘的连通区<br>    # 特别注意！！！，此操作会将整个图像也视为一个连通区域
    opening_x = opening.shape[0]
    opening_y = opening.shape[1]
    opening[:, 0] = 255
    opening[:, opening_y - 1] = 255
    opening[0, :] = 255
    opening[opening_x - 1, :] = 255
    # 检测图像连通区（输入为二值化图像）
    contours, hierarchy = cv2.findContours(opening, 1, 2)
    for n in range(len(contours)):
        # 筛选面积较大的连通区，阈值为20000
        cnt = contours[n]
        area = cv2.contourArea(cnt)
        if area > 20000:
            x, y, w, h = cv2.boundingRect(cnt)
            img_ = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 4)  # 画框
            print('')
            img_ = img_[y - h:y + h, x - w:x + w]
    cv2.imwrite(outputPath + img_name + 'abc_open.png' , opening)
    cv2.imwrite(outputPath + img_name + 'abc_close.png', closing)
    cv2.imwrite(outputPath + img_name + 'abc_close_range.png', img_)

def iterateImagePath(inputPath, outputPath):
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    if not os.path.exists(outputPath+ "/connectarea/"):
        os.mkdir(outputPath+ "/connectarea/")

    images = os.listdir(inputPath)
    for img_name in images:
        img = cv2.imread(inputPath + "/" + img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        connectArea(img, outputPath + "/connectarea/", img_name)

        #binary_img = Two_Pass(img, NEIGHBOR_HOODS_4)
        #binary_img, points = reorganize(binary_img)
        #outputPathFile = outputPath+ "/connectarea/"+img_name
        #cv2.imwrite(outputPathFile,binary_img)
        #print("save results to:", outputPathFile)

if __name__ == "__main__":
    print("Start to do image filtering with opencv")

    inputPath = "E:/ubuntushare/data/warehousetools01/segmentopencv/filter/medianblur/"
    # inputPath = "E:/ubuntushare/data/warehousetools01/original/"
    outputPath = "E:/ubuntushare/data/warehousetools01/segmentopencv/"

    iterateImagePath(inputPath, outputPath)
