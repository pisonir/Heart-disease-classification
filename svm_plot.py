from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
from decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

df = pd.read_csv('data/heart.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipe_svm = Pipeline([('scl', StandardScaler()),
                     ('pca', PCA(n_components=2)),
                     ('clf', SVC(kernel='rbf', random_state=1, C=0.1, gamma=0.01))])
pipe_svm.fit(X_train, y_train)
print('Train accuracy: %.3f' % pipe_svm.score(X_train, y_train))
print('Test accuracy: %.3f' % pipe_svm.score(X_test, y_test))
X_test_std = pipe_svm.named_steps['scl'].transform(X_test)
X_test_pca = pipe_svm.named_steps['pca'].transform(X_test_std)
plot_decision_regions(X=X_test_pca, y=y_test, classifier=pipe_svm.named_steps['clf'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.savefig('heart_PCA.png', dpi=500)