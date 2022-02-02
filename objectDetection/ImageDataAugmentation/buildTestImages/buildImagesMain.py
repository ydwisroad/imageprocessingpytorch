from jpgTopng import folderJPGtoPNG
from commonUtil import copyFilesWithSurfix
from labelme2voc import labelme2vocFormat
from CenterCrop import centerCropVOC
from CenterCrop import centerCropImagesOnly
from VOCOperation import vocResize
from trans_voc2yolo import voc2yolo
from trans_voc2yolo import transform2yolo
from mainImgAug import augmentImages
from combineAugmented import combineAugmentedAndYolo
from original2indexednames import original2indexednamesfolder

import configparser

def processRequests():
    config = configparser.ConfigParser()
    config.read("config.ini")
    for eachSectionName in config.sections():
        print("Going to handle ", eachSectionName)
        if eachSectionName == "General" and config.has_option(eachSectionName, "rootFolder"):
            rootFolder = config.get(eachSectionName, "rootFolder")
            continue

        options = config.options(eachSectionName)
        for eachOption in options:
            value = config.get(eachSectionName, eachOption)
            values = value.split(",")
            if len(values) == 2:
                inputFolderName  = values[0]
                outputFolderName = values[1]
                handleEachOperation(eachOption, rootFolder + inputFolderName + "/",
                                    rootFolder + outputFolderName + "/", rootFolder)
            elif len(values) == 3:
                inputFolderAName  = values[0]
                inputFolderBName = values[1]
                outputFolderName = values[2]
                handleEachOperation(eachOption, rootFolder + inputFolderAName + "/",
                                    rootFolder + outputFolderName + "/", rootFolder,
                                    rootFolder + inputFolderBName + "/",)
            else:
                continue

def handleEachOperation(operationName, inputFolder, outputFolder, rootFolder, inputFolderB=None):
    print("handle operation ", operationName + " inputFolder ", inputFolder, " ", outputFolder)
    if operationName == "jpgtopng":
        folderJPGtoPNG(inputFolder, outputFolder)
    if operationName == "copyjson":
        copyFilesWithSurfix(inputFolder, outputFolder, "json")
    if operationName == "labelme2voc":
        labelme2vocFormat(inputFolder, outputFolder)
    if operationName == "centercropvoc":
        centerCropVOC(inputFolder, outputFolder)
    if operationName == "vocresize":
        vocResize(inputFolder, outputFolder)
    if operationName == "voc2yolo":
        voc2yolo(inputFolder, outputFolder, rootFolder + "voc_classes.json")
    if operationName == "transform2yolo":
        transform2yolo(inputFolder, outputFolder)
    if operationName == "imageaugment":
        augmentImages(inputFolder, outputFolder, rootFolder)
    if operationName == "combineyoloaugment":
        combineAugmentedAndYolo(inputFolder, inputFolderB, outputFolder)
    if operationName == "centercropimagesonly":
        centerCropImagesOnly(inputFolder, outputFolder)

    if operationName == "original2indexednames":
        original2indexednamesfolder(inputFolder, outputFolder)

    return outputFolder

if __name__ == "__main__":
    print("Start to process the requests ")
    processRequests()





