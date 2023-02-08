import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

class BoostedTreeClassifier:
    def __init__(self, **kwargs):
        self.classifier = GradientBoostingClassifier()
        #self.classifier.loss = kwargs.get('loss_function', 'log_loss')
        self.classifier.learning_rate = kwargs.get('learning_rate', 0.1)
        self.classifier.max_depth = kwargs.get('max_depth', 3)
        self.classifier.n_estimators = kwargs.get('n_estimators', 200)



    def train(self, X, y):
        """
        Trains the classifier using the provided training data.
        :param X: 2D numpy array of shape (num_samples, num_features)
        :param y: 1D numpy array of shape (num_samples,)
        """
        self.classifier.fit(X, y)
        return

    def predict(self, XTe):
        """
        Predicts the binary class for each sample in the provided data.
        :param X: 2D numpy array of shape (num_samples, num_features)
        :return: 1D numpy array of shape (num_samples,)
        """
        return self.classifier.predict(XTe)

    def rank_features(self):
        feature_rank = deepcopy(self.classifier.feature_importances_)
        sorted_feature_rank_ind = np.argsort(feature_rank)
        return np.sort(feature_rank)
