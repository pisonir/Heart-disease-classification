from load_preprocessing_partition import load_partition
from knn_tuning_evaluation import tuning_evaluation

# Loading data and partitioning them in training and test datasets (80%/20% in this case).
X_train, X_test, y_train, y_test = load_partition(0.2)
# Selecting the best parameters for the KNN classifier
best_classifier = tuning_evaluation(X_train, y_train)
# Evaluate the classifier on the test data set
print('Test accuracy %.3f' % best_classifier.score(X_test, y_test))




