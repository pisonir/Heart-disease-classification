from load_preprocessing_partition import load_partition
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import numpy as np
np.random.seed(123)

# Loading data and partitioning them in training and test datasets (70%/30% in this case).
X_train, X_test, y_train, y_test = load_partition(0.2)
# Using Random Forest Classifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=5, random_state=10, n_jobs=-1)
forest.fit(X_train, y_train)
# Evaluate the classifier on the test data set
print('Test accuracy %.3f' % forest.score(X_test, y_test))

y_pred = forest.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
print('Sensitivity : ', sensitivity )

specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
print('Specificity : ', specificity)

y_train_str = y_train.astype('str')
y_train_str[y_train_str=='0'] = 'no disease'
y_train_str[y_train_str=='1'] = 'disease'
export_graphviz(forest.estimators_[3], out_file='tree.dot', feature_names=[i for i in X_train.columns], class_names=y_train_str.values,
                rounded=True, proportion=True, label='root', precision=2, filled=True)



