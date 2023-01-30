import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

class BoostedTreeClassifier:
    def __init__(self):
        self.classifier = GradientBoostingClassifier()

    def train(self, X, y):
        """
        Trains the classifier using the provided training data.
        :param X: 2D numpy array of shape (num_samples, num_features)
        :param y: 1D numpy array of shape (num_samples,)
        """
        self.classifier.fit(X, y)

    def predict(self, X):
        """
        Predicts the binary class for each sample in the provided data.
        :param X: 2D numpy array of shape (num_samples, num_features)
        :return: 1D numpy array of shape (num_samples,)
        """
        return self.classifier.predict(X)
