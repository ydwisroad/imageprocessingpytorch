import os

#Train Folder
trainFolder = '../../data/Crack_Segmentation_Dataset/val'

#Train Label Folder
trainLabelFolder = '../../data/Crack_Segmentation_Dataset/val_labels'

filesA = os.listdir(trainFolder)
filesB = os.listdir(trainLabelFolder)

for fileA in filesA:
    if (filesB.__contains__(fileA)!=True):
        print('labels not found ' + fileA)
        os.remove(trainFolder + '/' + fileA)

for fileB in filesB:
    if (filesA.__contains__(fileB)!=True):
        print('train not found ' + fileB)
        os.remove(trainLabelFolder+'/'+fileB)
print("End of the program")