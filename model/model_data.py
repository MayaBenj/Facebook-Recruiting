import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, mean_squared_error, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC


# Load data

train_data = pd.read_csv("../data_csv/train_with_features.csv")
x = train_data.iloc[:, 1:-1]
y = train_data.iloc[:, -1]

# Create polynomial features
poly = PolynomialFeatures(degree=1, include_bias=False)
polynomials = pd.DataFrame(poly.fit_transform(x))
x = pd.concat([x, polynomials], axis=1)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=5)

# Scale data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
"""
Logistic Regression

Polynomial features
# Using CV to calculate accuracy and choose degree
# CV score for d 3: 0.9321 (+/- 0.02)
# CV score for d 2: 0.9415 (+/- 0.02)
# CV score for d 1: 0.9503 (+/- 0.00)

Model data
Using CV to calculate accuracy and choose regularization coefficient
Accuracy for c 0.9: 0.8958 (+/- 0.05)
Accuracy for c 1.0: 0.8965 (+/- 0.05)
Accuracy for c 1.1: 0.8978 (+/- 0.05)
Accuracy for c 1.3: 0.8992 (+/- 0.05)
Accuracy for c 1.5: 0.8992 (+/- 0.05)
"""

# clf = LogisticRegression(fit_intercept=True, max_iter=10000, C=1.3)
# clf.fit(x_train_scaled, y_train)
# scores = cross_val_score(clf, x_train_scaled, y_train, cv=5)
# print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
# cost_train = mean_squared_error(y_train, clf.predict(x_train_scaled))
# cost_test = mean_squared_error(y_test, clf.predict(x_test_scaled))
# print("Cost train: %f, cost test: %f" % (cost_train, cost_test))
#
# train_score = clf.score(x_train_scaled, y_train)
# test_score = clf.score(x_test_scaled, y_test)
# print("Train score: %f, Test score: %f" % (train_score, test_score))
#
# f1_train = f1_score(y_train, (clf.predict_proba(x_train_scaled)[:, 1] >= 0.3).astype(int))
# f1_test = f1_score(y_test, (clf.predict_proba(x_test_scaled)[:, 1] >= 0.3).astype(int))
# print("F1-score for train %f, For test: %f" % (f1_train, f1_test))
#
# print("Top 4 most influential features: " + np.array_str(x.axes[1][np.argpartition(abs(clf.coef_), -4)[0, -4:]]))

"""
Random Forest
# Using CV to calculate accuracy and choose degree
# CV score for max_depth 20: 0.9483 (+/- 0.02)
# CV score for max_depth 15: 0.9503 (+/- 0.01)
# CV score for max_depth 10: 0.9523 (+/- 0.01)
# CV score for max_depth 5: 0.9476 (+/- 0.01)
# CV score for max_depth None: 0.9462 (+/- 0.01)

# CV score for min_samples_split 100: 0.9429 (+/- 0.01)
# CV score for min_samples_split 50: 0.9436 (+/- 0.01)
# CV score for min_samples_split 20: 0.9483 (+/- 0.01)
# CV score for min_samples_split 2: 0.9476 (+/- 0.01)

# CV score for min_samples_leaf 10: 0.9462 (+/- 0.01)
# CV score for min_samples_leaf 5: 0.9469 (+/- 0.01)
# CV score for min_samples_leaf 1: 0.9489 (+/- 0.01)

# CV score for max_leaf_nodes 20: 0.9469 (+/- 0.01)
# CV score for max_leaf_nodes 10: 0.9483 (+/- 0.01)
# CV score for max_leaf_nodes 5: 0.9456 (+/- 0.01)

# CV score for ccp_alpha  0.001: 0.9462 (+/- 0.02)
# CV score for ccp_alpha  0: 0.9503 (+/- 0.01)
"""

clf = RandomForestClassifier(max_depth=10, min_samples_split=20)
clf.fit(x_train_scaled, y_train)
scores = cross_val_score(clf, x_train_scaled, y_train, cv=5)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

cost_train = mean_squared_error(y_train, clf.predict(x_train_scaled))
cost_test = mean_squared_error(y_test, clf.predict(x_test_scaled))
print("Cost train: %f, cost test: %f" % (cost_train, cost_test))

train_score = clf.score(x_train_scaled, y_train)
test_score = clf.score(x_test_scaled, y_test)
print("Train score: %f, Test score: %f" % (train_score, test_score))

f1_train = f1_score(y_train, (clf.predict_proba(x_train_scaled)[:, 1] >= 0.3).astype(int))
f1_test = f1_score(y_test, (clf.predict_proba(x_test_scaled)[:, 1] >= 0.3).astype(int))
print("F1-score for train %f, For test: %f" % (f1_train, f1_test))

print("Top 4 most influential features: " + np.array_str(x.axes[1][np.argpartition(clf.feature_importances_, -4)[-4:]]))

"""
Support Vector Machine

# CV score for C
# CV score for c 2: 0.9422 (+/- 0.01)
# CV score for c 1: 0.9462 (+/- 0.00)

# CV score for kernel
# CV score for kernel linear: 0.9449 (+/- 0.00)
# CV score for poly linear: 0.9415 (+/- 0.01)
# CV score for poly sigmoid: 0.9281 (+/- 0.02)
# CV score for poly rbf: 0.9462 (+/- 0.00)

"""

# clf = SVC(probability=True)
# clf.fit(x_train_scaled, y_train)
# scores = cross_val_score(clf, x_train_scaled, y_train, cv=5)
# print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
# cost_train = mean_squared_error(y_train, clf.predict(x_train_scaled))
# cost_test = mean_squared_error(y_test, clf.predict(x_test_scaled))
# print("Cost train: %f, cost test: %f" % (cost_train, cost_test))
#
# train_score = clf.score(x_train_scaled, y_train)
# test_score = clf.score(x_test_scaled, y_test)
# print("Train score: %f, Test score: %f" % (train_score, test_score))
#
# f1_train = f1_score(y_train, (clf.predict_proba(x_train_scaled)[:, 1] >= 0.3).astype(int))
# f1_test = f1_score(y_test, (clf.predict_proba(x_test_scaled)[:, 1] >= 0.3).astype(int))
# print("F1-score for train %f, For test: %f" % (f1_train, f1_test))


"""
K-Nearest Neighbors

# CV score for n_neighbors
# CV score for n_neighbors 5: 0.9388 (+/- 0.02)
# CV score for n_neighbors 10: 0.9442 (+/- 0.01)
# CV score for n_neighbors 20: 0.9462 (+/- 0.00)

"""

# clf = KNeighborsClassifier(n_neighbors=20)
# clf.fit(x_train_scaled, y_train)
# scores = cross_val_score(clf, x_train_scaled, y_train, cv=5)
# print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
# cost_train = mean_squared_error(y_train, clf.predict(x_train_scaled))
# cost_test = mean_squared_error(y_test, clf.predict(x_test_scaled))
# print("Cost train: %f, cost test: %f" % (cost_train, cost_test))
#
# train_score = clf.score(x_train_scaled, y_train)
# test_score = clf.score(x_test_scaled, y_test)
# print("Train score: %f, Test score: %f" % (train_score, test_score))
#
# f1_train = f1_score(y_train, (clf.predict_proba(x_train_scaled)[:, 1] >= 0.3).astype(int))
# f1_test = f1_score(y_test, (clf.predict_proba(x_test_scaled)[:, 1] >= 0.3).astype(int))
# print("F1-score for train %f, For test: %f" % (f1_train, f1_test))


# Apply model to test data
test_data = pd.read_csv("../data_csv/test_with_features.csv")
X = test_data.iloc[:, 1:]
poly = PolynomialFeatures(degree=1, include_bias=False)
X_scaled = pd.DataFrame(scaler.transform(X))
polynomials = pd.DataFrame(poly.fit_transform(X_scaled))
X_scaled = pd.concat([X_scaled, polynomials], axis=1)
Y = pd.Series(clf.predict_proba(X_scaled)[:, 1])
Y.name = "prediction"
prediction_data = pd.concat([test_data.bidder_id, Y], axis=1)
# Add users with no data as 0 (humans)
test_data_original = pd.read_csv("../data_csv/test.csv")
no_data_users = test_data_original[~test_data_original.bidder_id.isin(test_data.bidder_id)][["bidder_id"]]
no_data_users["prediction"] = 0
prediction_data = prediction_data.append(no_data_users)
# Create prediction csv
prediction_data.to_csv("../data_csv/prediction.csv", index=False)


