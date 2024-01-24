import numpy as np
from statistics import mean
from naiveBayesClassifier import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from ucimlrepo import fetch_ucirepo

iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

classifier = NaiveBayesClassifier()
accuracies = []
precisions = []
recalls = []
f1s = []
conf_matrices = []

for state in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=state
    )

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train["class"].values
    y_test = y_test["class"].values

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    accuracies.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred, average="weighted"))
    recalls.append(recall_score(y_test, y_pred, average="weighted"))
    f1s.append(f1_score(y_test, y_pred, average="weighted"))
    conf_matrices.append(confusion_matrix(y_test, y_pred))

print("Average Confusion Matrix:")
print(np.mean(conf_matrices, axis=0))
print(f"Average accuracy: {mean(accuracies):.4f}")
print(f"Average precision: {mean(precisions):.4f}")
print(f"Average recall: {mean(recalls):.4f}")
print(f"Average f1 score: {mean(f1s):.4f}")
