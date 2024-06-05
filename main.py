import numpy as np
from sklearn import svm,naive_bayes,ensemble,linear_model
from sklearn.metrics import accuracy_score
import time
import pickle
import pygame
import pandas as pd
import random
import tkinter as tk
from tkinter import ttk

filename = "ApneaData.pkl"
f = open(filename,'rb')
data = pickle.load(f)
f.close()

features = []
classes = []

for row in data:
    features.append(row[:-1])
    classes.append(row[-1])

inputLength = len(features)
testLength = int(inputLength*0.2)
train_features, train_classes=features[:-testLength], classes[:-testLength]
test_features,test_classes = features[-testLength:],classes[-testLength:]
t=time.time()
print("preprocessing time:",(time.time()-t))

clf=ensemble.RandomForestClassifier(n_estimators=30)
clf.fit(train_features,train_classes)
print("fitting time:",(time.time()-t))
t=time.time()

pred_classes = clf.predict(test_features)

for i, pred_class in enumerate(pred_classes[:6]):
    if pred_class == 1:
        print(f"Sleep apnea detected for test data point {i+1}!")
        # Play a sound alert
        pygame.mixer.init()
        pygame.mixer.music.load("alert.mp3.wav")
        pygame.mixer.music.play()
        input("Press any key to continue...")
    else:
        print(f"No sleep apnea detected for test data point {i+1}.")

pred_classes=[]
for e in test_features:
    pred_classes.append(clf.predict([e])[0])
score = accuracy_score(pred_classes,test_classes)*100
print("predicting time:",(time.time()-t))
print("Accuracy:",score)