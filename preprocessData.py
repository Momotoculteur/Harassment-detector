#IMPORT
import pandas as pd
from sklearn.utils import shuffle


def cleanResultData():
    '''
    Je me suis rendu compte bien trop tard que dans mon dataset, j'avais des fois des soucis de crawl
    et je me retrouve donc sur certaines de mes phrases avec une classe non conforme (autre que 0 ou 1)
    Cette fonction va permettre de corriger tout ça
    :return:
    '''

data = pd.read_csv('dataset\\data.txt', sep=',', names=["text", "result"])

list = ['1','0']
for index, row in data.iterrows():
    if row['result'] not in list:
        #print(row['result'])
        data.drop(index,inplace=True)



print(data['result'].value_counts())
data.to_csv('dataset\\data.txt', header=None, index=None, sep=',', mode='w')