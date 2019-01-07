#IMPORT
import pandas as pd
from sklearn.utils import shuffle


def adaptDatasetTest():
    '''
    Permet de me cut un dataset de test equilibre et melange
    :return:
    '''

data = pd.read_csv('.\\datasetTest\\dataTest.txt', sep=',', names=["text", "result"])
print(data['result'].value_counts())

max1 = 0
max0 = 0

list = ['1','0']
for index, row in data.iterrows():
    if row['result'] == 0:
        if max0 < 50:
            max0 = max0+1
        else:
            data.drop(index, inplace=True)
    if row['result'] == 1:
        if max1 < 50:
            max1 = max1 + 1
        else:
            data.drop(index, inplace=True)
        #data.drop(index,inplace=True)


print(data['result'].value_counts())
data = shuffle(data)
#print(data['result'].value_counts())
data.to_csv('datasetTest\\dataTest.txt', header=None, index=None, sep=',', mode='w')


def cleanResultData():
    '''
    Je me suis rendu compte bien trop tard que dans mon dataset, j'avais des fois des soucis de crawl
    et je me retrouve donc sur certaines de mes phrases avec une classe non conforme (autre que 0 ou 1)
    Cette fonction va permettre de corriger tout ça
    :return:
    '''

data = pd.read_csv('datasetTest\\dataTest.txt', sep=',', names=["text", "result"])
print(data['result'].value_counts())

list = ['1','0']
for index, row in data.iterrows():
    if row['result'] not in list:
        #print(row['result'])
        data.drop(index,inplace=True)



print(data['result'].value_counts())
data.to_csv('datasetTest\\dataTest.txt', header=None, index=None, sep=',', mode='w')


if __name__ == "__main__":
    """
    # MAIN
    """
    cleanResultData()
