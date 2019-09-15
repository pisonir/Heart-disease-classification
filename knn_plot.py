from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pandas as pd
from decision_regions import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as KNN

df = pd.read_csv('data/heart.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipe_knn = Pipeline([('scl', StandardScaler()),
                     ('pca', PCA(n_components=2)),
                     ('clf', KNN(n_neighbors=50, p=2, metric='minkowski'))])
pipe_knn.fit(X_train, y_train)
print('Train accuracy: %.3f' % pipe_knn.score(X_train, y_train))
print('Test accuracy: %.3f' % pipe_knn.score(X_test, y_test))
X_test_std = pipe_knn.named_steps['scl'].transform(X_test)
X_test_pca = pipe_knn.named_steps['pca'].transform(X_test_std)
plot_decision_regions(X=X_test_pca, y=y_test, classifier=pipe_knn.named_steps['clf'])
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc='lower left')
plt.savefig('heart_KNN.png', dpi=500)
