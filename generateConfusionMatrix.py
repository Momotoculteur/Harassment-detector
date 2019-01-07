# IMPORT
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
from keras.models import load_model
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing import text, sequence
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from keras import utils


"""
# Classe permettant de génerer une matrice de confusion à partir d'un dataset de test et d'un modèle entrainé
# au préalable
"""


def generateMatrix(model, datasetTestPath, destinationMatrix):
    max_words = 10000

    data = pd.read_csv(datasetTestPath, sep=',', names=["text", "result"])
    tokenize = text.Tokenizer(num_words=max_words, char_level=False)
    testText = data['text']
    testResult = data['result']

    tokenize.fit_on_texts(testText)
    xTest = tokenize.texts_to_matrix(testText)

    encoder = LabelEncoder()
    encoder.fit(testResult)
    y_softmax = model.predict(xTest)

    yTest = encoder.transform(testResult)
    num_classes = np.max(yTest) + 1
    yTest = utils.to_categorical(yTest, num_classes)
    y_test_1d = []
    y_pred_1d = []

    for i in range(len(yTest)):
        probs = yTest[i]
        index_arr = np.nonzero(probs)
        one_hot_index = index_arr[0].item(0)
        y_test_1d.append(one_hot_index)

    for i in range(0, len(y_softmax)):
        probs = y_softmax[i]
        predicted_index = np.argmax(probs)
        y_pred_1d.append(predicted_index)

    text_labels = encoder.classes_
    cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    plt.figure(figsize=(24,20))
    plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
    plt.savefig(destinationMatrix + '\\MatriceConfusion')


def plot_confusion_matrix(cm, classes, normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    # cm = cm.astype('int64') / cm.sum(axis=1)[:, np.newaxis]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main():
	"""
	# Fonction main
	"""

	#On definit les chemins de nos divers ressources
	modelPath = '.\\modelTrained\\model.hdf5'
	datasetTestPath = '.\\datasetTest\\dataTest.txt'
	destinationMatrix = '.\\graph'
	model = load_model(modelPath)

	generateMatrix(model, datasetTestPath, destinationMatrix)


if __name__ == "__main__":
	"""
	# MAIN
	"""
	main()