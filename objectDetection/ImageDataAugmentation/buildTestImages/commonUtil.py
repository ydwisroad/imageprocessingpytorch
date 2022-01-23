import os
import shutil
from glob import glob

def copyFilesWithSurfix(source_path, target_path, surfix):
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    files = glob(source_path + "*."+ surfix)
    files = [i.replace("\\", "/").split("/")[-1] for i in files]

    for file_ in files:
        filename = source_path + file_
        shutil.copy(filename, target_path + file_)



if __name__ == "__main__":
    print("start of the program")