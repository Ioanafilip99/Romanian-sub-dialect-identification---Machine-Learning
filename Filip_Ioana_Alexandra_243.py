import io
import nltk
#nltk.download("punkt")
import numpy as np
from numpy import set_printoptions, nan
set_printoptions(threshold=nltk.sys.maxsize)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


# CITIREA DATELOR

test_samples = np.genfromtxt("data/test_samples.txt", encoding="utf-8", dtype=None, delimiter="\t", names=("key", "text"), comments = None)
train_samples = np.genfromtxt("data/train_samples.txt", encoding="utf-8", dtype=None, delimiter="\t", names=("key", "text"), comments = None)
train_labels = np.genfromtxt("data/train_labels.txt", encoding="utf-8", dtype=None, delimiter="\t", names=("key", "dialect"), comments = None)
validation_samples = np.genfromtxt("data/validation_samples.txt", encoding="utf-8", dtype=None, delimiter="\t", names=("key", "text"), comments = None)
validation_labels = np.genfromtxt("data/validation_labels.txt", encoding="utf-8", dtype=None, delimiter="\t", names=("key", "dialect"), comments = None)

# PRELUAREA DATELOR NECESARE

# date de test
test_samples_keys = test_samples["key"]
test_samples_texts = test_samples["text"]
# date de antrenare
train_samples_texts = train_samples["text"]
train_labels_dialects = train_labels["dialect"]
# date de validare
validation_labels_dialects = validation_labels["dialect"].tolist()
validation_samples_texts = validation_samples["text"].tolist()

# ANTRENAREA SI VALIDAREA

# CountVectorizer care imparte textul in cuvinte
# ce contine cuvintele si nr de aparitii in text
count_vect = CountVectorizer()

train_data_matrix = count_vect.fit_transform(train_samples_texts)
validation_data_matrix = count_vect.transform(validation_samples_texts)


MultinomialNB_classifier = MultinomialNB()
MultinomialNB_classifier.fit(train_data_matrix, train_labels_dialects)

# predictii
MultinomialNB_classifier_predictions_train = MultinomialNB_classifier.predict(validation_data_matrix)

# acuratetea
accuracy = round(accuracy_score(validation_labels_dialects, MultinomialNB_classifier_predictions_train)*100, 2)
print("MultinomialNB Accuracy: ", accuracy, "%")

# scorul f1
f1 = round(f1_score(validation_labels_dialects, MultinomialNB_classifier_predictions_train)*100, 2)
print("MultinomialNB f1: ", f1, "%")

# raportul clasificarii
print("Classification Report: \n" , classification_report(validation_labels_dialects, MultinomialNB_classifier_predictions_train))

# matricea de confuzie
print("Confusion matrix: \n", confusion_matrix(validation_labels_dialects, MultinomialNB_classifier_predictions_train))

# TESTAREA

test_data_matrix = count_vect.transform(test_samples_texts)
MultinomialNB_classifier_predictions_test = MultinomialNB_classifier.predict(test_data_matrix)

# SCRIEREA FISIERULUI DE PREDICTII

with io.open("predictions.txt", "a+", encoding="utf8") as pred_txt:
    pred_txt.write("id,label\n")
    for pred_idx in range(len(MultinomialNB_classifier_predictions_test)):
        key = str(test_samples_keys[pred_idx])
        dialect = str(MultinomialNB_classifier_predictions_test[pred_idx])
        pred_txt.write(key+","+dialect +"\n")