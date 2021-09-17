import numpy as np
import csv
import shutil

def extractPartCSV(csvPath, outputCSVPath, lines):
    rows = []
    with open(csvPath) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        #print(rows)
    f.close()

    fOut = open(outputCSVPath, 'w', encoding='utf-8')
    csv_writer = csv.writer(fOut)

    for i in range(0, lines):
        csv_writer.writerow(rows[i])

    fOut.close()

def copyImages(csvFile, imagesSourceFolder, imagesDestFolder):
    print("copy images")
    rows = []
    with open(csvFile) as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
    f.close()

    for eachRow in rows:
        if (eachRow[1].find("jpg") > -1):
            sourceFullFile = imagesSourceFolder + eachRow[1]
            destFullFile = imagesDestFolder+eachRow[1]
            print("source full file ", sourceFullFile + " " + destFullFile)

            shutil.copyfile(sourceFullFile, destFullFile)

if __name__ == "__main__":
    print("Start of generate Shopee simple dataset")
    csvPath = "../../../data/shopee/simple/train.csv"
    outputCSVPath = "../../../data/shopee/simple/trainSimple.csv"

    extractPartCSV(csvPath, outputCSVPath, 1000)

    copyImages(outputCSVPath, "../../../data/shopee/train_images/", "../../../data/shopee/simple/train_images/")

