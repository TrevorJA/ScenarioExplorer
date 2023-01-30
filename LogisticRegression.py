import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticRegressionClassifier:
    def __init__(self):
        self.classifier = LogisticRegression()

    def train(self, X, y):
        """
        Trains the classifier using the provided training data.
        :param X: 2D numpy array of shape (num_samples, num_features)
        :param y: 1D numpy array of shape (num_samples,)
        """
        self.classifier.fit(X, y)

    def predict(self, X):
        """
        Predicts the class for each sample in the provided data.
        :param X: 2D numpy array of shape (num_samples, num_features)
        :return: 1D numpy array of shape (num_samples,)
        """
        return self.classifier.predict(X)
