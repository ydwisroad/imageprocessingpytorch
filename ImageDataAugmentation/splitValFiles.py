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


#5,10,15,16,17,27,30,35,37,42,43,45,48,49,50,52,54,55,56,57
#0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17,18,19
def replaceIndexInLabels(labelsFolder):
    changeDic = {'5':0, '10':1, '15':2, '16':3, '17':4, '27':5, '30':6, '35':7, '37':8, '42':9, '43':10, '45':11, '48':12, '49':13, '50':14, '52':15, '54':16, '55':17, '56':18, '57':19}
    for fileName in os.listdir(labelsFolder):
        eachLabelFile = labelsFolder + "/" + fileName
        print("eachFile ", eachLabelFile)
        with open(eachLabelFile, "r") as f1, open(eachLabelFile +".bak", "w") as f2:
            for line in f1:
                lineItems = line.split(" ")
                firstOne = lineItems[0]
                #print(" find key ", firstOne)
                replacedOne = changeDic[str(firstOne)]
                line = line.replace(str(firstOne) + " ", str(replacedOne) + " ")
                f2.write(line)
                continue

        os.remove(eachLabelFile)
        os.rename(eachLabelFile +".bak", eachLabelFile)

def renamePNGtoJPG(folder):
    for fileName in os.listdir(folder):
        eachFile = folder + "/" + fileName
        newFileName = fileName.replace("png", "jpg")
        newFile = folder + "/" + newFileName
        os.rename(eachFile, newFile)


if __name__ == "__main__":
    #copyToValFolder("/Users/i052090/Downloads/segmentation/data/TSRD/full/train/images",
    #                "/Users/i052090/Downloads/segmentation/data/TSRD/full/val/")
    #moveLabelFileToRightFolder("/Users/i052090/Downloads/segmentation/data/TSRD/full/train/images",
    #                           "/Users/i052090/Downloads/segmentation/data/TSRD/full/train/labels")

    #selectSmallObjects([5,10,15,16,17,27,30,35,37,42,43,45,48,49,50,52,54,55,56,57],
    #                   "/Users/i052090/Downloads/segmentation/data/TSRD/newOutput/allSmallObjects",
    #                   "/Users/i052090/Downloads/segmentation/data/TSRD/newOutput/select20smallobjects")

    #replaceIndexInLabels("/Users/i052090/Downloads/segmentation/data/TSRD/twenty/train/labelstest/")
    folder = "/Users/i052090/Downloads/roadproject/marks/yolo/all/images"
    renamePNGtoJPG(folder)

