import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm._libsvm import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from chapter03.utilities import visualize_classifier


# Argument parser
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Classify data using Ensemble Learning techniques')
    parser.add_argument('--classifier-type', dest='classifier_type', required=True, choices=['rf', 'erf'],
                        help='Type of classifier to use;can be either "rf" or "erf"')

    return parser


if __name__ == "__main__":
    # Parse the input argument
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type

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

    # Ensemble Learning classifier
    params = {
        'n_estimators': 100,
        'max_depth': 4,
        'random_state': 0
    }

    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)
    elif classifier_type == 'erf':
        classifier = ExtraTreesClassifier(**params)
    else:
        raise ValueError("The classifier_type must be 'rf' or 'erf'")

    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train, 'Training dataset')
    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test, 'Test dataset')

    # Evaluate classifier performance
    class_names = ['Class-0', 'Class-1', 'Class-2']
    print('\n' + '#' * 40)
    print('\nClassifier performance on training dataset \n')
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print('#' * 40 + '\n')
    print('#' * 40 + '\n')
    print('\nClassifier performance on test dataset\n')
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print('#' * 40 + '\n')
