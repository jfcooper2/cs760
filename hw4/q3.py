import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import re

lettermap    = {l:i for i, l in enumerate("abcdefghijklmnopqrstuvwxyz ")}
letterinvmap = {i:l for i, l in enumerate("abcdefghijklmnopqrstuvwxyz ")}

files = os.listdir("data")
filedata = {}
bowdata = {}
letterdata = {}
for filename in files:
    with open("data/" + filename, "r") as infile:
        filedata[filename] = re.sub(r'[^a-zA-Z ]+', '', infile.read()).lower()

        letters = np.zeros(27) + 0.5 # Additive Smoothing
        bow = np.zeros(27)
        for letter in filedata[filename]:
            letters[lettermap[letter]] += 1
            bow[lettermap[letter]] += 1
    
        bowdata[filename] = bow
        letterdata[filename] = letters / np.sum(letters)

langmap = {'e': 0, 's': 1, 'j': 2}
priors = np.zeros(3) + 0.5 
langprobs = np.zeros((3,27))

trainletterdata = {}
testletterdata = {}

for key in letterdata.keys():
    lang = key[0]
    if int(key[1:-4]) < 10:
        trainletterdata[key] = letterdata[key]
        priors[langmap[lang]] += 1
        langprobs[langmap[lang]] += letterdata[key]
    else:
        testletterdata[key] = letterdata[key]
for i in range(3):
    langprobs[i] /= (priors[i] - 0.5)
priors /= np.sum(priors)

priors = np.log(priors)

print("PRIORS:", np.exp(priors))
print("LANG PRORS")
print("e:")
print(langprobs[0])
print("s:")
print(langprobs[1])
print("j:")
print(langprobs[2])
print()

print("E10:")
print(bowdata["e10.txt"])
print()

logpxe = 0
logpxs = 0
logpxj = 0
for letter in range(27):
    logpxe += bowdata["e10.txt"][letter] * np.log(langprobs[0,letter])
    logpxs += bowdata["e10.txt"][letter] * np.log(langprobs[1,letter])
    logpxj += bowdata["e10.txt"][letter] * np.log(langprobs[2,letter])
print("E10: log p(x|e):", logpxe, "log p(x|s):", logpxs, "log p(x|j):", logpxj)
print()

# Bayes rule : log(p(e|x)) = log(p(x|e)) + log(p(e)) - log(p(x))
logpx = 0
logpex = logpxe + priors[0]
logpsx = logpxs + priors[1]
logpjx = logpxj + priors[2]
"""
logpx = logpex + logpsx + logpjx
logpex -= logpx
logpsx -= logpx
logpjx -= logpx
"""

print("E10: log p(x|e):", logpex, "log p(x|s):", logpsx, "log p(x|j):", logpjx)

labels = np.zeros(len(testletterdata.keys()))
predictions = np.zeros(len(testletterdata.keys()))
for index, key in enumerate(testletterdata.keys()):
    labels[index] = langmap[key[0]]

    logpxe = 0
    logpxs = 0
    logpxj = 0
    for letter in range(27):
        logpxe += bowdata[key][letter] * np.log(langprobs[0,letter])
        logpxs += bowdata[key][letter] * np.log(langprobs[1,letter])
        logpxj += bowdata[key][letter] * np.log(langprobs[2,letter])
    
    # Bayes rule : log(p(e|x)) = log(p(x|e)) + log(p(e)) - log(p(x))
    logpx = 0
    logpex = logpxe + priors[0]
    logpsx = logpxs + priors[1]
    logpjx = logpxj + priors[2]

    if logpex > logpsx:
        if logpjx > logpex:
            predictions[index] = langmap['j']
        else:
            predictions[index] = langmap['e']
    else: # ex < sx
        if logpjx > logpsx:
            predictions[index] = langmap['j']
        else:
            predictions[index] = langmap['s']

print(labels)
print(predictions)

matrix = confusion_matrix(labels, predictions)
disp = ConfusionMatrixDisplay(matrix, display_labels=["English", "Spanish", "Japanese"])
disp.plot()
plt.title("Confusion Matrix - q3p7")
plt.savefig("img/q3p7.png")
plt.show()
