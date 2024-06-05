from flask import Flask, render_template, jsonify
from sklearn.decomposition import PCA
import numpy as np
from sklearn import svm, naive_bayes, ensemble, linear_model
from sklearn.metrics import accuracy_score
import time
import pickle
import pygame
import pandas as pd
import random
import tkinter as tk
from tkinter import ttk

app = Flask(__name__)

# Load the data from main.py
filename = "ApneaData.pkl"
f = open(filename, 'rb')
data = pickle.load(f)
f.close()

features = []
classes = []
for row in data:
    features.append(row[:-1])
    classes.append(row[-1])

inputLength = len(features)
testLength = int(inputLength * 0.2)
train_features, train_classes = features[:-testLength], classes[:-testLength]
test_features, test_classes = features[-testLength:], classes[-testLength:]
t = time.time()
preprocessing_time = (time.time() - t)

clf = ensemble.RandomForestClassifier(n_estimators=30)
clf.fit(train_features, train_classes)
fitting_time = (time.time() - t)

# Predict the classes of the test data
pred_classes = clf.predict(test_features)

# Check if sleep apnea is detected for the first 6 test data points
test_results = []
for i, pred_class in enumerate(pred_classes[:6]):
    if pred_class == 1:
        test_results.append(f"Sleep apnea detected for test data point {i+1}!")
    else:
        test_results.append(f"No sleep apnea detected for test data point {i+1}.")
        time.sleep(0.5)

pred_classes = []
for e in test_features:
    pred_classes.append(clf.predict([e])[0])
predicting_time = (time.time() - t)
accuracy = accuracy_score(pred_classes, test_classes) * 100

@app.route('/')
def index():
    return render_template('index.html', test_results=test_results, preprocessing_time=preprocessing_time, fitting_time=fitting_time, predicting_time=predicting_time, accuracy=accuracy)

@app.route('/data')
def data():
    # Generate the plots and data points outputs from datavis.py
    filename = "ApneaData.pkl"
    features = []
    classes = []

    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()

    for row in data:
        features.append(row[:-1])
        classes.append(row[-1])

    # Convert features list to numpy array
    features = np.array(features)

    pca = PCA(n_components=3)
    reduced_features = pca.fit_transform(features)

    # Create a boolean mask to separate the positive and negative cases
    positive_mask = np.array(classes) == 1
    negative_mask = np.array(classes) == 0

    # Calculate the percentage of variance explained by each principal component
    pca_variance = pca.explained_variance_ratio_

    # Identify the most important features
    most_important_features = np.argsort(pca.components_, axis=1)[:, -3:]

    # Analyze the separation between positive and negative cases
    positive_centroid = reduced_features[positive_mask].mean(axis=0)
    negative_centroid = reduced_features[negative_mask].mean(axis=0)
    distance = np.linalg.norm(positive_centroid - negative_centroid)

    return jsonify({
        'reduced_features': reduced_features.tolist(),
        'positive_mask': positive_mask.tolist(),
        'negative_mask': negative_mask.tolist(),
        'pca_variance': pca_variance.tolist(),
        'most_important_features': most_important_features.tolist(),
        'positive_centroid': positive_centroid.tolist(),
        'negative_centroid': negative_centroid.tolist(),
        'distance': distance
    })

if __name__ == '__main__':
    app.run(debug=True)