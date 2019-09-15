from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data/heart.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
pca = PCA(n_components=2)
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.fit_transform(X_test_std)
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3,
                              random_state=0)
ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=0)
x_min = X_test_pca[:,0].min() - 1
x_max = X_test_pca[:,0].max() + 1
y_min = X_test_pca[:,1].min() - 1
y_max = X_test_pca[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max,0.1))
ada.fit(X_train_pca, y_train)
Z = ada.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_test_pca[y_test==0,0], X_test_pca[y_test==0,1], c='blue', marker='^')
plt.scatter(X_test_pca[y_test==1,0], X_test_pca[y_test==1,1], c='red', marker='o')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('AdaBoost.png', dpi=500, bbox_inches='tight')

