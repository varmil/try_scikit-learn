import numpy
import pandas as pd
from sklearn import svm
from sklearn import datasets
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
clf = svm.SVC(C=1.0, kernel='rbf')

# dummy dataset
# iris = datasets.load_iris()
# X, y = iris.data, iris.target

# real dataset
dataFrame = pd.read_csv("./csv/ticker-5min.csv")
# X = dataFrame.weighted_average5.as_matrix()
X = dataFrame.weighted_average10.as_matrix()
y = dataFrame.next_result.as_matrix()

# preapare (scaling)
X = X.reshape(-1, 1)
X = preprocessing.scale(X)

# create test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=39)

# grid search
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
parameters = [
    # 'linear', 'rbf'
    {'kernel':['rbf'], 'C':numpy.logspace(-4, 4, 9), 'gamma':numpy.logspace(-4, 4, 9)}
]
clf = GridSearchCV(clf, parameters, cv=4, n_jobs = -1)

# learning
clf.fit(X_train, y_train)

# estimate
result = clf.predict(X_test)

# print("\n+ トレーニングデータでCVした時の平均スコア:\n")
print("Grid scores on development set:")
print()
for params, mean_score, scores in clf.grid_scores_:
    print("%0.3f (+/-%0.03f) for %r"
          % (mean_score, scores.std() * 2, params))
print()

# result
print(clf.best_estimator_)
print("best score is " + str(clf.best_score_))
print(classification_report(y_test, result))
# print(confusion_matrix(y_test, result))
