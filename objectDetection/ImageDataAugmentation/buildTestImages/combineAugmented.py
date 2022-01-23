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

def combineAugmentedAndYolo(originalYolo, augmentedYolo, outputYolo):
    print("Combine original yolo and original ")
    if not os.path.exists(outputYolo):
        os.mkdir(outputYolo)
        os.chmod(outputYolo,stat.S_IWOTH)
    if not os.path.exists(outputYolo + "/images"):
        os.mkdir(outputYolo + "/images")
    if not os.path.exists(outputYolo + "/images/train"):
        os.mkdir(outputYolo + "/images/train")
    if not os.path.exists(outputYolo + "/images/val"):
        os.mkdir(outputYolo + "/images/val")

    if not os.path.exists(outputYolo + "/labels"):
        os.mkdir(outputYolo + "/labels")
    if not os.path.exists(outputYolo + "/labels/train"):
        os.mkdir(outputYolo + "/labels/train")
    if not os.path.exists(outputYolo + "/labels/val"):
        os.mkdir(outputYolo + "/labels/val")

    copyFilesFromFolder(originalYolo + "/images/train", outputYolo + "/images/train/", "png")
    copyFilesFromFolder(originalYolo + "/labels/train", outputYolo + "/labels/train/", "txt")

    copyFilesFromFolder(originalYolo + "/images/val", outputYolo + "/images/val/", "png")
    copyFilesFromFolder(originalYolo + "/labels/val", outputYolo + "/labels/val/", "txt")

    total_files = glob.glob(augmentedYolo + "/*.txt")
    total_files = [i.replace("\\", "/").split("/")[-1].split(".txt")[0] for i in total_files]
    train_files, val_files = train_test_split(total_files, test_size=0.2, random_state=4)

    for file in train_files:
        shutil.copy(augmentedYolo + "/" + file +".png", outputYolo + "/images/train/")
        shutil.copy(augmentedYolo + "/" + file +".txt", outputYolo + "/labels/train/")

    for file in val_files:
        shutil.copy(augmentedYolo + "/" + file +".png", outputYolo + "/images/val/")
        shutil.copy(augmentedYolo + "/" + file +".txt", outputYolo + "/labels/val/")

if __name__ == "__main__":
    originalYolo = rootDir + "yolo"
    augmentedYolo = rootDir + "augmented"

    outputYolo = rootDir + "outputYolo"
    combineAugmentedAndYolo(originalYolo, augmentedYolo, outputYolo)
