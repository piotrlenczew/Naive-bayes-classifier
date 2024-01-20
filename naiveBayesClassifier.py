import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.classes = {}
        self.means = {}
        self.stds = {}
        self.class_probs = {}

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = X_c.mean(axis=0)
            self.stds[c] = X_c.std(axis=0)
            self.class_probs[c] = len(X_c) / len(X)

    def _calculate_probability(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def _predict_instance(self, x):
        probabilities = {}

        for c in self.classes:
            class_prob = np.log(self.class_probs[c])
            feature_probs = np.sum(np.log(self._calculate_probability(x[i], self.means[c][i], self.stds[c][i]))
                                   for i in range(len(x)))
            probabilities[c] = class_prob + feature_probs

        return max(probabilities, key=probabilities.get)

    def predict(self, X):
        return [self._predict_instance(x) for x in X]
