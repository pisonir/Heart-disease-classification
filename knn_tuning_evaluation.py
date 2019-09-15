from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier as KNN


def tuning_evaluation(X_train,y_train):
    '''
    Grid search algorithm to tune hyperparameters and evaluation of a accuracy. Returns the best classifier.
    :param X_train: numpy array [n_samples, n_features]
    :param y_train: numpy array [n_samples]
    :return: best classifier.
    '''

    pipe_KNN = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=2)),
                         ('clf', KNN(p=2, metric='minkowski'))])
    param_range = [1, 2, 4, 6, 8, 10, 20, 30, 50, 100]
    gs = GridSearchCV(estimator = pipe_KNN,
                      param_grid = [{'clf__n_neighbors': param_range}],
                      scoring = 'accuracy',
                      cv = 10,
                      n_jobs = 1)
    gs = gs.fit(X_train, y_train)
    print('The grid search best score is ',gs.best_score_)
    print('The best parameters according to the grid search algorithm are ',gs.best_params_)

    return gs.best_estimator_



