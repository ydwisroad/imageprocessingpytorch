from imgUtils import *
from augmentImages import *
from Helpers import *
from util import *
import os
import shutil
import glob


def copyToValFolder(allFilesPath, valFilesPath):
    iCount = 0
    for imgFileName in glob.glob(os.path.join(allFilesPath, "*.png")):
        #Copy to validation folder
        if (iCount % 8 == 2):
            imgFullPath =  imgFileName
            labelFullPath =  imgFileName.replace("png","txt")

            shutil.copy(imgFullPath, valFilesPath + "/images")
            shutil.copy(labelFullPath, valFilesPath + "/labels")

        iCount = iCount + 1


def moveLabelFileToRightFolder(allFilesPath, realDest):
    for labelFileName in glob.glob(os.path.join(allFilesPath, "*.txt")):
        shutil.move(labelFileName, realDest)

def selectSmallObjects(selectedIds, originalSmallObjectsFolder, outputSmallObjFolder):
    for fileName in os.listdir(os.path.join(originalSmallObjectsFolder)):
        print("file Name ", fileName)
        clsPrefix = fileName.split("_")[0]
        if (int(clsPrefix) in selectedIds):
            fullNamePath = originalSmallObjectsFolder + "/" + fileName
            print("selected ", fullNamePath)
            shutil.copy(fullNamePath, outputSmallObjFolder)



if __name__ == "__main__":
    #copyToValFolder("/Users/i052090/Downloads/segmentation/data/TSRD/full/train/images",
    #                "/Users/i052090/Downloads/segmentation/data/TSRD/full/val/")
    #moveLabelFileToRightFolder("/Users/i052090/Downloads/segmentation/data/TSRD/full/train/images",
    #                           "/Users/i052090/Downloads/segmentation/data/TSRD/full/train/labels")

    selectSmallObjects([5,10,15,16,17,27,30,35,37,42,43,45,48,49,50,52,54,55,56,57],
                       "/Users/i052090/Downloads/segmentation/data/TSRD/newOutput/allSmallObjects",
                       "/Users/i052090/Downloads/segmentation/data/TSRD/newOutput/select20smallobjects")
