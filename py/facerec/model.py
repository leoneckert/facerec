#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) Philipp Wagner. All rights reserved.
# Licensed under the BSD license. See LICENSE file in the project root for full license information.

from facerec.feature import AbstractFeature
from facerec.classifier import AbstractClassifier

class PredictableModel(object):
    def __init__(self, feature, classifier, dimensions=None, namesDict=None):
        if not isinstance(feature, AbstractFeature):
            raise TypeError("feature must be of type AbstractFeature!")
        if not isinstance(classifier, AbstractClassifier):
            raise TypeError("classifier must be of type AbstractClassifier!")
        
        self.feature = feature
        self.classifier = classifier
        self.dimensions = dimensions
        self.namesDict = namesDict
    
    def compute(self, X, y):
        features = self.feature.compute(X,y)
        self.classifier.compute(features,y)

    def predict(self, X):
        q = self.feature.extract(X)
        pred_idx = self.classifier.predict(q)[0]
        pred_name = self.namesDict[pred_idx]       
        prediction = self.classifier.predict(q)
        prediction[1]["name"] = pred_name
        return prediction
        
    def __repr__(self):
        feature_repr = repr(self.feature)
        classifier_repr = repr(self.classifier)
        dimensions_repr = repr(self.dimensions)
        namesDict_repr = repr(self.namesDict)
        return "PredictableModel (feature=%s, classifier=%s, dimensions=%s, namesDict=%s)" % (feature_repr, classifier_repr, dimensions_repr, )
