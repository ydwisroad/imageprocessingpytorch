import os, sys, stat
import shutil
import glob
import tqdm


rootDir = "E:/ubuntushare/data/warehousetools/"

def copyFilesFromFolder(source, dest, suffix):
    for file in glob.glob(source + '/**/*.' + suffix, recursive=True):
        file = file.replace("\\","/")
        shutil.copyfile(file, dest + file[(file.rfind("/")+1):])

def combineAugmentedAndOriginal(originalYolo, augmentedYolo, outputYolo):
    print("Combine original yolo and original ")
    if not os.path.exists(outputYolo):
        os.mkdir(outputYolo)
        os.chmod(outputYolo,stat.S_IWOTH)
    if not os.path.exists(outputYolo + "/images"):
        os.mkdir(outputYolo + "/images")
        os.chmod(outputYolo+ "/images", stat.S_IWOTH)
    if not os.path.exists(outputYolo + "/labels"):
        os.mkdir(outputYolo + "/labels")
        os.chmod(outputYolo + "/labels", stat.S_IWOTH)

    copyFilesFromFolder(originalYolo, outputYolo + "/images/", "png")
    copyFilesFromFolder(originalYolo, outputYolo + "/labels/", "txt")

    copyFilesFromFolder(augmentedYolo, outputYolo + "/images/", "png")
    copyFilesFromFolder(augmentedYolo, outputYolo + "/labels/", "txt")

if __name__ == "__main__":
    originalYolo = rootDir + "yolo"
    augmentedYolo = rootDir + "augmented"
    outputYolo = rootDir + "outputYolo"
    combineAugmentedAndOriginal(originalYolo, augmentedYolo, outputYolo)
