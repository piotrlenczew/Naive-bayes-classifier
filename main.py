from naiveBayesClassifier import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
iris = fetch_ucirepo(id=53)

X = iris.data.features
y = iris.data.targets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.values
X_test = X_test.values
y_train = y_train['class'].values
y_test = y_test.values

classifier = NaiveBayesClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

accuracy_custom = accuracy_score(y_test, y_pred)
print(f'Accuracy (Custom Naive Bayes): {accuracy_custom}')
