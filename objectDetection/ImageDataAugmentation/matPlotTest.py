import pandas as pd
import matplotlib.pyplot as plt
import csv

plt.rcParams['font.sans-serif']=['SimHei']

#pip install xlrd
def plotIncomeTest():
    datafile = u'./incomeData.xls'
    data = pd.read_excel(datafile)

    print("time ", data["Time"])

    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    plt.title('All Income Line ', fontsize=20)  # 标题，并设定字号大小
    plt.xlabel(u'x-year', fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel(u'y-income', fontsize=14)  # 设置y轴，并设定字号大小

    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    plt.plot(data['Time'], data['Jay'], color="deeppink", linewidth=2, linestyle=':', label='Jay income', marker='o')
    plt.plot(data['Time'], data['Jolin'], color="darkblue", linewidth=1, linestyle='--', label='Jolin income', marker='+')
    plt.plot(data['Time'], data['Justin'], color="goldenrod", linewidth=1.5, linestyle='-', label='Justin income', marker='*')

    plt.legend(loc=2)  #
    plt.savefig("./allIncomeChart.png", dpi=200)
    #plt.show()  #

def convertSpaceToCommaToCSV():
    with open('./results.csv') as infile, open('./resultsNew.csv', 'w') as outfile:
        for line in infile:
            outfile.write(" ".join(line.split()).replace(' ', ','))
            outfile.write(",")  # trailing comma shouldn't matter
            outfile.write('\n')
    #input = open('./resultsNew.csv', 'r')
    #output = open('resultsNew2.csv', 'w')

    #for row in input:
    #    row = row.strip()
    #    if len(row) != 0:
    #        output.write(row)
    #        output.write('\n')
    #input.close()
    #output.close()

#Epoch,Memory,trainBoxLoss,trainObjLoss,trainClsLoss,trainLoss,count,fixed,precision,recall,map05,map0595,valBoxLoss,valObjLoss,valClsLoss

def plotTSTrainLossResult():
    datafile = u'./resultsNew.xls'
    data = pd.read_excel(datafile)

    #print("time ", data["Time"])

    plt.figure(figsize=(12, 5))  # 设置画布的尺寸
    plt.title('训练损失', fontsize=10)  # 标题，并设定字号大小
    plt.xlabel(u'代', fontsize=12)     # 设置x轴，并设定字号大小
    plt.ylabel(u'损失', fontsize=12)      # 设置y轴，并设定字号大小

    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    #, marker = 'o' , marker='+'  , marker='*'
    plt.plot(data['Epoch'], data['trainBoxLoss'], color="green", linewidth=1, linestyle='solid', label='边界框损失')
    plt.plot(data['Epoch'], data['trainObjLoss'], color="darkblue", linewidth=1, linestyle=':', label='目标损失')
    plt.plot(data['Epoch'], data['trainClsLoss'], color="goldenrod", linewidth=1, linestyle='-.', label='类别损失')

    plt.legend(loc=1)  #
    plt.savefig("./resultsTrainLoss.png", dpi=200)

def plotTSValLossResult():
    datafile = u'./resultsNew.xls'
    data = pd.read_excel(datafile)

    #print("time ", data["Time"])

    plt.figure(figsize=(12, 5))  # 设置画布的尺寸
    plt.title('验证损失', fontsize=10)  # 标题，并设定字号大小
    plt.xlabel(u'代', fontsize=12)     # 设置x轴，并设定字号大小
    plt.ylabel(u'损失', fontsize=12)      # 设置y轴，并设定字号大小

    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    #, marker = 'o' , marker='+'  , marker='*'
    plt.plot(data['Epoch'], data['valBoxLoss'], color="green", linewidth=1, linestyle='solid', label='边界框损失')
    plt.plot(data['Epoch'], data['valObjLoss'], color="darkblue", linewidth=1, linestyle=':', label='目标损失')
    plt.plot(data['Epoch'], data['valClsLoss'], color="goldenrod", linewidth=1, linestyle='-.', label='类别损失')

    plt.legend(loc=1)  #
    plt.savefig("./resultsValLoss.png", dpi=200)

def plotPrecisionRecallResult():
    datafile = u'./resultsNew.xls'
    data = pd.read_excel(datafile)

    #print("time ", data["Time"])

    plt.figure(figsize=(12, 5))  # 设置画布的尺寸
    plt.title('指标:精确度/召回率', fontsize=10)  # 标题，并设定字号大小
    plt.xlabel(u'代', fontsize=12)     # 设置x轴，并设定字号大小
    plt.ylabel(u'精确度/召回率', fontsize=12)      # 设置y轴，并设定字号大小

    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    #, marker = 'o' , marker='+'  , marker='*'
    plt.plot(data['Epoch'], data['precision'], color="green", linewidth=1, linestyle=':', label='精确度')
    plt.plot(data['Epoch'], data['recall'], color="darkblue", linewidth=1, linestyle='-', label='召回率')

    plt.legend(loc=4)  #
    plt.savefig("./resultsPrecisionRecall.png", dpi=200)

def plotMetricsMapResult():
    datafile = u'./resultsNew.xls'
    data = pd.read_excel(datafile)

    # print("time ", data["Time"])

    plt.figure(figsize=(12, 5))  # 设置画布的尺寸
    plt.title('指标:mAP_0.5/mAP_0.5:95', fontsize=10)  # 标题，并设定字号大小
    plt.xlabel(u'代', fontsize=12)  # 设置x轴，并设定字号大小
    plt.ylabel(u'mAP_0.5/mAP_0.5:95', fontsize=12)  # 设置y轴，并设定字号大小

    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    # , marker = 'o' , marker='+'  , marker='*'
    plt.plot(data['Epoch'], data['map05'], color="green", linewidth=1, linestyle=':', label='mAP_0.5')
    plt.plot(data['Epoch'], data['map0595'], color="darkblue", linewidth=1, linestyle='-', label='mAP_0.5:95')

    plt.legend(loc=4)  #
    plt.savefig("./resultsmAP050595.png", dpi=200)

if __name__ == "__main__":
    #plotIncomeTest()
    plotTSTrainLossResult()
    plotTSValLossResult()
    plotPrecisionRecallResult()
    plotMetricsMapResult()







