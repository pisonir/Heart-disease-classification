from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def tuning_evaluation(X_train,y_train):
    '''
    Grid search algorithm to tune hyperparameters and evaluation of a accuracy. Returns the best classifier.
    :param X_train: numpy array [n_samples, n_features]
    :param y_train: numpy array [n_samples]
    :return: best classifier.
    '''

    pipe_svm = Pipeline([('scl', StandardScaler()),
                         ('pca', PCA(n_components=2)),
                         ('clf', SVC(random_state=1))])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'clf__C': param_range,
                   'clf__kernel': ['linear']},
                  {'clf__C': param_range,
                   'clf__gamma': param_range,
                   'clf__kernel': ['rbf']}]
    gs = GridSearchCV(estimator = pipe_svm,
                      param_grid = param_grid,
                      scoring = 'accuracy',
                      cv = 10,
                      n_jobs = 1)
    gs = gs.fit(X_train, y_train)
    print('The grid search best score is ',gs.best_score_)
    print('The best parameters according to the grid search algorithm are ',gs.best_params_)

    return gs.best_estimator_



