from torchreid.utils import FeatureExtractor
import scipy.io as scio
from glob import glob
import numpy as np
from torch.nn import functional as F
from torchreid import metrics
import torch

def loadExtractor(model_name, model_path, device = 'cpu'):
    extractor = FeatureExtractor(
        model_name=model_name,
        model_path=model_path,
        device=device
    )
    return extractor

def extractFeaturesFromFiles(extractor, image_list):
    features = extractor(image_list)
    return features

def getImageListFromFolder(imageFolder):
    files = glob(imageFolder + "*.jpg")
    if len(files) == 0:
        files = glob(imageFolder + "*.png")
    imagesList = [i.replace("\\", "/").split("/")[-1] for i in files]
    imagesList = [imageFolder + "/" + eachImage for eachImage in imagesList]

    return imagesList

#https://cloud.tencent.com/developer/article/1390390 load write MAT
def saveMatrixToMat(featuresMatrix, dataMatPath):
    npArrayFeatures = np.array(featuresMatrix)
    scio.savemat(dataMatPath, {'queryFeatures': npArrayFeatures})

def loadMatDataFromFile(dataMatPath):
    data = scio.loadmat(dataMatPath)
    return data

if __name__ == '__main__':
    print("Start to extract features")
    model_name = 'osnet_x1_0'
    model_path = 'E:/ubuntushare/gputrain/REID/result/Feb2_tools/model/model.pth.tar-250'

    imagesQueryList = getImageListFromFolder('E:/ubuntushare/data/Market1501/market1501/query/')
    imagesGalleryList = getImageListFromFolder('E:/ubuntushare/data/Market1501/market1501/bounding_box_test/')

    extractor = loadExtractor(model_name, model_path)
    featuresQuery = extractFeaturesFromFiles(extractor, imagesQueryList)
    featuresGallery = extractFeaturesFromFiles(extractor, imagesGalleryList)
    #saveMatrixToMat("./dataNew.mat", featuresQuery)

    #featuresQuery = F.normalize(featuresQuery, p=2, dim=1)
    #featuresGallery = F.normalize(featuresGallery, p=2, dim=1)
    distmat = metrics.compute_distance_matrix(featuresQuery, featuresGallery, "cosine")
    print("Calculation metrics done distmat ", distmat)
    newdistOrdered, indices = distmat.sort(descending=True)
    print("Calculation metrics done newdistOrdered ", newdistOrdered)
    print("Calculation metrics done indices ", indices)
    x = torch.randn(3, 4)
    print("x original ", x)
    sorted, indicesA = x.sort(descending=True)
    print("x sorted ", sorted)
    print("indices ", indicesA)


