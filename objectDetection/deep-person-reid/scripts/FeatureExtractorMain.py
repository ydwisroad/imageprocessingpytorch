from torchreid.utils import FeatureExtractor
import scipy.io as scio
from glob import glob
import numpy as np

def extractFeaturesFromFiles(model_name, model_path, image_list, device = 'cpu'):
    extractor = FeatureExtractor(
        model_name=model_name,
        model_path=model_path,
        device=device
    )
    features = extractor(image_list)

    return features

def getImageListFromFolder(imageFolder):
    files = glob(imageFolder + "*.jpg")
    imagesList = [i.replace("\\", "/").split("/")[-1] for i in files]

    imagesList = [imageFolder + "/" + eachImage for eachImage in imagesList]

    return imagesList


if __name__ == '__main__':
    print("Start to extract features")
    model_name = 'osnet_x1_0'
    model_path = 'E:/ubuntushare/gputrain/REID/result/Jan30/log/osnet_x1_0_market1501_softmax_cosinelr/model/model.pth.tar-250'
    imagesList = getImageListFromFolder('E:/ubuntushare/data/Market1501/market1501/querysmall/')

    features = extractFeaturesFromFiles(model_name, model_path, imagesList)
    npArrayFeatures = np.array(features)

    #https://cloud.tencent.com/developer/article/1390390 load write MAT
    dataNew = './dataNew.mat'
    scio.savemat(dataNew, {'queryFeatures': npArrayFeatures})

    data = scio.loadmat(dataNew)
    print("data ", data)