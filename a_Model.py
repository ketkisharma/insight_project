import pandas as pd
import cPickle

from flaskexample import predictor

def ModelIt(title, body):
    predictions = predictor.predict(title, body)
    aboveThreshold =  [word for _, score, word in predictions if score >= 0.9]
    toReturn = []
    if len(aboveThreshold) < 10:
        toReturn = [word for _, _, word in predictions[:10]]
    else:
        toReturn = aboveThreshold
    return toReturn
