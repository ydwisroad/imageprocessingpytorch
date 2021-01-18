from imgUtils import *
from augmentImages import *
from Helpers import *
from util import *


if __name__ == "__main__":
    print("This is the start of main program of main image augmentation")

    #addLetterBoxesForFolder("/Users/i052090/Downloads/segmentation/data/TSRDMini/train/images",
    #                        "/Users/i052090/Downloads/segmentation/data/TSRDMini/train/labels",
    #                        "/Users/i052090/Downloads/segmentation/data/TSRDMini/letterboxed/images",
    #                        "/Users/i052090/Downloads/segmentation/data/TSRDMini/letterboxed/labels",
    #                        [512,512],[0,0,0])

    #cropObjectsFromImagePath("/Users/i052090/Downloads/segmentation/data/TSRD/train/images/",
    #                         "/Users/i052090/Downloads/segmentation/data/TSRD/train/labels/",
    #                         "/Users/i052090/Downloads/segmentation/data/TSRD/objects")

    #copySmallObjectsToOneBlankImage("/Users/i052090/Downloads/segmentation/data/TSRD/newOutput/trafficsign.png",
    #                                "/Users/i052090/Downloads/segmentation/data/TSRD/objects",
    #                                "/Users/i052090/Downloads/segmentation/data/TSRD/newOutput/images",
    #                                8, 20000)

    #generateAugmentedObjects("/Users/i052090/Downloads/segmentation/data/TSRD/objects",
    #                         "/Users/i052090/Downloads/segmentation/data/TSRD/newOutput/smallObjects")
    #copySmallObjectsToOneBlankImage("/Users/i052090/Downloads/segmentation/data/TSRD/newOutput/trafficsign.png",
    #                                "/Users/i052090/Downloads/segmentation/data/TSRD/newOutput/select20smallobjects",
    #                                "/Users/i052090/Downloads/segmentation/data/TSRD/twenty/train/images",
    #                                9, 15000)

    addObjRectToImages("/Users/i052090/Downloads/segmentation/data/TSRD/twenty/train/images/",
                       "/Users/i052090/Downloads/segmentation/data/TSRD/twenty/train/labels",
                       "/Users/i052090/Downloads/segmentation/data/TSRD/twenty/train/annonated")










