from os import scandir
import os
import shutil

def original2indexednamesfolder(inputFolder, outputFolder):
    print("This is the start of original2indexednamesfolder")
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if not os.path.exists(outputFolder + "/working"):
        os.makedirs(outputFolder + "/working")

    folders = scandir(inputFolder)
    for item in folders:
        if item.is_dir():
            imagesItems = scandir(item)
            for eachImage in imagesItems:
                destImage = outputFolder + "/working/" + item.name + "_c1s1_" + eachImage.name
                print("dest Image ", destImage)
                shutil.copyfile(eachImage.path,
                                destImage)

if __name__ == '__main__':
    print("This is the start of original images to indexed names images")
    inputFolder = "E:/ubuntushare/data/Market1501/tools0202/0202"
    outputFolder = "E:/ubuntushare/data/Market1501/tools0202/indexed"

    original2indexednamesfolder(inputFolder, outputFolder)
