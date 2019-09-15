from sklearn.model_selection import train_test_split
import pandas as pd

pd.options.mode.chained_assignment = None  #hide any pandas warnings

def load_partition(test_size):
    '''
    Return features and labels partitioned in training and test data sets.
    :param test_size: int.
    :return: X_train, X_test numpy arrays of shape [n_samples, n_features],
    y_train, y_test numpy array of shape [n_samples]
    '''

    df = pd.read_csv('data/heart.csv')
    df.columns = ['age', 'sex', 'chest_pain', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar',
                  'rest_ecg_measurements', 'maximum_heart_rate', 'ex_induced_angina',
                  'ST_depression', 'ST_slope', 'num_major_vessels', 'thalassemia', 'target']
    df['sex'][df['sex'] == 0] = 'female'
    df['sex'][df['sex'] == 1] = 'male'

    df['chest_pain'][df['chest_pain'] == 1] = 'typical angina'
    df['chest_pain'][df['chest_pain'] == 2] = 'atypical angina'
    df['chest_pain'][df['chest_pain'] == 3] = 'non-anginal pain'
    df['chest_pain'][df['chest_pain'] == 4] = 'asymptomatic'

    df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
    df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

    df['rest_ecg_measurements'][df['rest_ecg_measurements'] == 0] = 'normal'
    df['rest_ecg_measurements'][df['rest_ecg_measurements'] == 1] = 'ST-T wave abnormality'
    df['rest_ecg_measurements'][df['rest_ecg_measurements'] == 2] = 'left ventricular hypertrophy'

    df['ex_induced_angina'][df['ex_induced_angina'] == 0] = 'no'
    df['ex_induced_angina'][df['ex_induced_angina'] == 1] = 'yes'

    df['ST_slope'][df['ST_slope'] == 1] = 'upsloping'
    df['ST_slope'][df['ST_slope'] == 2] = 'flat'
    df['ST_slope'][df['ST_slope'] == 3] = 'downsloping'

    df['thalassemia'][df['thalassemia'] == 1] = 'normal'
    df['thalassemia'][df['thalassemia'] == 2] = 'fixed defect'
    df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'

    df = pd.get_dummies(df, drop_first=True) #drop the first category of each so to have e.g. 'male' = 0 or 1

    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='target'), df['target'], test_size=test_size, random_state=10)

    return X_train, X_test, y_train, y_test




