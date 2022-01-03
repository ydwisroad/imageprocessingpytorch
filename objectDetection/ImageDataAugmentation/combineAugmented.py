import os, sys, stat
import shutil
import glob
import tqdm
from sklearn.model_selection import train_test_split

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
    if not os.path.exists(outputYolo + "/train"):
        os.mkdir(outputYolo + "/train")
    if not os.path.exists(outputYolo + "/train/images"):
        os.mkdir(outputYolo + "/train/images")
    if not os.path.exists(outputYolo + "/train/labels"):
        os.mkdir(outputYolo + "/train/labels")

    if not os.path.exists(outputYolo + "/val"):
        os.mkdir(outputYolo + "/val")
    if not os.path.exists(outputYolo + "/val/images"):
        os.mkdir(outputYolo + "/val/images")
    if not os.path.exists(outputYolo + "/val/labels"):
        os.mkdir(outputYolo + "/val/labels")

    copyFilesFromFolder(originalYolo + "/train", outputYolo + "/train/images/", "png")
    copyFilesFromFolder(originalYolo + "/train", outputYolo + "/train/labels/", "txt")

    copyFilesFromFolder(originalYolo + "/val", outputYolo + "/val/images/", "png")
    copyFilesFromFolder(originalYolo + "/val", outputYolo + "/val/labels/", "txt")

    total_files = glob.glob(augmentedYolo + "/*.txt")
    total_files = [i.replace("\\", "/").split("/")[-1].split(".txt")[0] for i in total_files]
    train_files, val_files = train_test_split(total_files, test_size=0.2, random_state=4)

    for file in train_files:
        shutil.copy(augmentedYolo + "/" + file +".png", outputYolo + "/train/images/")
        shutil.copy(augmentedYolo + "/" + file +".txt", outputYolo + "/train/labels/")

    for file in val_files:
        shutil.copy(augmentedYolo + "/" + file +".png", outputYolo + "/val/images/")
        shutil.copy(augmentedYolo + "/" + file +".txt", outputYolo + "/val/labels/")

    #copyFilesFromFolder(augmentedYolo, outputYolo + "/train/images/", "png")
    #copyFilesFromFolder(augmentedYolo, outputYolo + "/train/labels/", "txt")

if __name__ == "__main__":
    originalYolo = rootDir + "yolo"
    augmentedYolo = rootDir + "augmented"

    outputYolo = rootDir + "outputYolo"
    combineAugmentedAndOriginal(originalYolo, augmentedYolo, outputYolo)
