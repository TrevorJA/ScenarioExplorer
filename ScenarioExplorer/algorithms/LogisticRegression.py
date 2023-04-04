import numpy as np
from sklearn.linear_model import LogisticRegression

class LogisticRegressionClassifier:
    def __init__(self, **kwargs):
        self.classifier = LogisticRegression()
        self.max_iterations = kwargs.get('max_iterations', 300)


    def train(self, X, y):
        """
        Trains the classifier using the provided training data.
        :param X: 2D numpy array of shape (num_samples, num_features)
        :param y: 1D numpy array of shape (num_samples,)
        """

        self.classifier.max_iter = self.max_iterations
        self.classifier.fit(X, y)
        return

    def predict(self, XTe):
        """
        Predicts the class for each sample in the provided data.
        :param X: 2D numpy array of shape (num_samples, num_features)
        :return: 1D numpy array of shape (num_samples,)
        """
        return self.classifier.predict_proba(XTe)[:, 1]

    def get_params(self):
        """
        Returns the parameters for the model.
        """
        return self.classifier.get_params()
