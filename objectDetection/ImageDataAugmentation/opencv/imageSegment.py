import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

#https://blog.csdn.net/qq_52309640/article/details/120941157
def thresholdSegment(img2, outputImage):
    #img0 = cv2.imread(inputImage)
    #img1 = cv2.resize(img0, dsize=None, fx=0.5, fy=0.5)
    #img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #print(h, w)
    #cv2.namedWindow("W0")
    #cv2.imshow("W0", img2)
    #cv2.waitKey(delay=0)
    # 图像进行二值化
    ##第一种阈值类型
    ret0, img3 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
    #print(ret0)
    ##第二种阈值类型
    ret1, img4 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)
    #print(ret1)
    ##第三种阈值类型
    ret2, img5 = cv2.threshold(img2, 127, 255, cv2.THRESH_TRUNC)
    #print(ret2)
    ##第四种阈值类型
    ret3, img6 = cv2.threshold(img2, 127, 255, cv2.THRESH_TOZERO)
    #print(ret3)
    ##第五种阈值类型
    ret4, img7 = cv2.threshold(img2, 127, 255, cv2.THRESH_TOZERO)
    #print(ret4)
    # 将所有阈值类型得到的图像绘制到同一张图中
    plt.rcParams['font.family'] = 'SimHei'  # 将全局中文字体改为黑体
    figure = [img2, img3, img4, img5, img6, img7]
    title = ["原图", "第一种阈值类型", "第二种阈值类型", "第三种阈值类型", "第四种阈值类型", "第五种阈值类型"]
    for i in range(6):
        figure[i] = cv2.cvtColor(figure[i], cv2.COLOR_BGR2RGB)  # 转化图像通道顺序，这一个步骤要记得
        plt.subplot(3, 2, i + 1)
        plt.imshow(figure[i])
        plt.title(title[i])  # 添加标题
    plt.savefig(outputImage)  # 保存图像，如果不想保存也可删去这一行
    print("save results to ", outputImage)
    #plt.show()

#https://blog.csdn.net/qq_52309640/article/details/120940908?spm=1001.2014.3001.5501
def sobel(img, outputImage):
    img3 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    cv2.imwrite(outputImage, img3)
    print("save results to ", outputImage)

def laplacian(img, outputImage):
    img3 = cv2.Laplacian(img, cv2.CV_64F)
    cv2.imwrite(outputImage, img3)
    print("save results to ", outputImage)

def scharr(img, outputImage):
    img3 = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    cv2.imwrite(outputImage, img3)
    print("save results to ", outputImage)

def canny(img, outputImage):
    img3 = cv2.Canny(img, 100, 200)
    cv2.imwrite(outputImage, img3)
    print("save results to ", outputImage)

#https://blog.csdn.net/u010429424/article/details/73692907
def watershed(img, outputImage):
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #cv2.imshow("thresh", thresh)
    cv2.imwrite(outputImage + "thresh.jpg", thresh)

    # Step3. 对图像进行“开运算”，先腐蚀再膨胀
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    #cv2.imshow("opening", opening)
    cv2.imwrite(outputImage + "opening.jpg", opening)

    # Step4. 对“开运算”的结果进行膨胀，得到大部分都是背景的区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    #cv2.imshow("sure_bg", sure_bg)
    cv2.imwrite(outputImage + "sure_bg.jpg", sure_bg)

    # Step5.通过distanceTransform获取前景区域
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
    #cv2.imshow("sure_fg", sure_fg)
    cv2.imwrite(outputImage + "sure_bg2.jpg", sure_bg)

    # Step6. sure_bg与sure_fg相减,得到既有前景又有背景的重合区域
    sure_fg = np.uint8(sure_fg)
    unknow = cv2.subtract(sure_bg, sure_fg)

    # Step7. 连通区域处理
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknow == 255] = 0

    # Step8.分水岭算法
    #markers = cv2.watershed(img, markers)
    #img[markers == -1] = [0, 255, 0]

    #cv2.imwrite(outputImage, img)
    #print("save results to ", outputImage)

def grubCut(img, outputImage):
    mask = np.zeros(img.shape[:2], np.uint8)  # 创建大小相同的掩模
    bgdModel = np.zeros((1, 65), np.float64)  # 创建背景图像
    fgdModel = np.zeros((1, 65), np.float64)  # 创建前景图像

    # Step3. 初始化矩形区域
    # 这个矩形必须完全包含前景（相当于这里的梅西）
    rect = (50, 50, 450, 290)

    # Step4. GrubCut算法，迭代5次
    # mask的取值为0,1,2,3
    #img = img.astype(np.uint8)  No use

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)  # 迭代5次

    # Step5. mask中，值为2和0的统一转化为0, 1和3转化为1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    cv2.imwrite(outputImage, img)
    print("save results to ", outputImage)

def kmeans(img1, outputImage):
    Z = img1.reshape((-1, 3))
    Z = np.float32(Z)  # 转化数据类型
    c = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 4  # 聚类中心个数，一般来说也代表聚类后的图像中的颜色的种类
    ret, label, center = cv2.kmeans(Z, k, None, c, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    img9 = res.reshape((img1.shape))
    cv2.imwrite(outputImage, img9)
    print("save results to ", outputImage)

def waterfen(imgPath, outputImage):
    img0 = cv2.imread(imgPath)
    img1 = cv2.resize(img0, dsize=None, fx=0.5, fy=0.5)
    img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret1, img10 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # （图像阈值分割，将背景设为黑色）
    cv2.imwrite(outputImage + "threshold.jpg", img10)
    ##noise removal（去除噪声，使用图像形态学的开操作，先腐蚀后膨胀）
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(img10, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imwrite(outputImage + "opening.jpg", opening)
    # sure background area(确定背景图像，使用膨胀操作)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    cv2.imwrite(outputImage + "sure_bgdilate.jpg", sure_bg)

    # Finding sure foreground area（确定前景图像，也就是目标）
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    cv2.imwrite(outputImage + "sure_fgtransform.jpg", sure_bg)
    # Finding unknown region（找到未知的区域）
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labelling
    ret3, markers = cv2.connectedComponents(sure_fg)  # 用0标记所有背景像素点
    # Add one to all labels so that sure background is not 0, but 1（将背景设为1）
    markers = markers + 1
    ##Now, mark the region of unknown with zero（将未知区域设为0）
    markers[unknown == 255] = 0
    markers = cv2.watershed(img1, markers)  # 进行分水岭操作
    img1[markers == -1] = [0, 0, 255]

    cv2.imwrite(outputImage, img1)
    print("save results to ", outputImage)

def iterateImagePath(inputPath, outputPath):
    print("interate for path ", inputPath)
    if not os.path.exists(outputPath):
        os.mkdir(outputPath)
    images = os.listdir(inputPath)

    if not os.path.exists(outputPath + "/threshold/"):
        os.mkdir(outputPath + "/threshold/")
    if not os.path.exists(outputPath + "/sobel/"):
        os.mkdir(outputPath + "/sobel/")
    if not os.path.exists(outputPath + "/laplacian/"):
        os.mkdir(outputPath + "/laplacian/")
    if not os.path.exists(outputPath + "/scharr/"):
        os.mkdir(outputPath + "/scharr/")
    if not os.path.exists(outputPath + "/canny/"):
        os.mkdir(outputPath + "/canny/")
    if not os.path.exists(outputPath + "/watershed/"):
        os.mkdir(outputPath + "/watershed/")
    if not os.path.exists(outputPath + "/grubCut/"):
        os.mkdir(outputPath + "/grubCut/")
    if not os.path.exists(outputPath + "/kmeans/"):
        os.mkdir(outputPath + "/kmeans/")
    if not os.path.exists(outputPath + "/waterfen/"):
        os.mkdir(outputPath + "/waterfen/")

    for img_name in images:
        img0 = cv2.imread(inputPath + "/" + img_name)
        img1 = cv2.resize(img0, dsize=None, fx=0.5, fy=0.5)
        img2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        #thresholdSegment(img2,outputPath+"/threshold/" + img_name)
        #sobel(img2,outputPath+"/sobel/" + img_name)
        #laplacian(img2,outputPath+"/laplacian/" + img_name)
        #scharr(img2,outputPath+"/scharr/" + img_name)
        #canny(img2, outputPath + "/canny/" + img_name)
        #watershed(img2, outputPath + "/watershed/" + img_name)
        #grubCut(img2, outputPath + "/grubCut/" + img_name)
        #kmeans(img2, outputPath + "/kmeans/" + img_name)
        waterfen(inputPath + "/" + img_name, outputPath + "/waterfen/" + img_name)


if __name__ == "__main__":
    print("Start to do image segmentation with opencv")
    inputPath = "E:/ubuntushare/data/warehousetools01/testsegment/"
    outputPath = "E:/ubuntushare/data/warehousetools01/segmentopencv/"
    iterateImagePath(inputPath, outputPath)
