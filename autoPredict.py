#IMPORT
from keras.models import load_model
import time
from keras.preprocessing import text
import os


def predict(modelPath, maxWord):

    # Chargement du modele
    print("\nChargement du modèle...")
    model = load_model(modelPath)

    while True :
        os.system('cls')
        # Recuperer le texte à tester
        inputText = input("\nEntrez votre texte : ")

        start = time.time()

        print("\nTokenization du texte...")
        # Transforme le text vers une matrice de mot et permet de lui donnée un indice
        tokenize = text.Tokenizer(num_words=maxWord, char_level=False)
        tokenize.fit_on_texts(inputText)
        word = tokenize.texts_to_matrix(inputText)

        print("\nPrediction du texte...")
        prediction = model.predict(word)[0]
        #predictionWithLabel = text_labels[np.argmax(prediction)]
        end = time.time()
        print("\nProbabilites (temps : {0:.2f}secs)".format(end-start))
        print("\t- Non harcelement : {0:.2f}%".format(prediction[0]*100.))
        print("\t- Harcelement : {0:.2f}%".format(prediction[1]*100.))





        again = input("\nRecommencer ? (O/N) ")
        if again == 'n':
            return


if __name__ == "__main__":
    """
    # MAIN
    """
    modelPath = '.\\modelTrained\\model.hdf5'
    maxWord = 10000

    predict(modelPath, maxWord)