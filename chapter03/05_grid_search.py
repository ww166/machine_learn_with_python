# Finding optimal training parameters using grid search
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm._libsvm import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from chapter03.utilities import visualize_classifier

# Load input data
input_file = 'data/data_random_forests.txt'
# Read data
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Separate input data into two classes based on labels
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])

# Visualize input data
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white', edgecolors='black', linewidths=1, marker='s')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', edgecolors='black', linewidths=1, marker='o')
plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white', edgecolors='black', linewidths=1, marker='^')
plt.title('Input data')

plt.show()

# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Define the parameter grid
parameter_grid = [
    {
        'n_estimators': [100],
        'max_depth': [2, 4, 7, 12, 16]
    },
    {
        'max_depth': [4, ],
        'n_estimators': [25, 50, 100, 250]
    }
]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print('\nSearching optimal parameters for ', metric)
    classifier = GridSearchCV(ExtraTreesClassifier(random_state=0),
                              param_grid=parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    print('\nGrid scores for the parameter grid: ')

    # 此处函数更换 grid_scores_ 更换为 cv_results_,并且变更为字典了。
    # for params, avg_score, _ in classifier.grid_scores_:
    for k, v in classifier.cv_results_.items():
        # print(k, '-->', round(v, 3))
        print(k, '-->', v)

    print('\nBest parameters: ', classifier.best_params_)

    y_pred = classifier.predict(X_test)
    print('\nPerformance report: \n')
    print(classification_report(y_test, y_pred))