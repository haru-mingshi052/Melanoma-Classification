import pandas as pd
import numpy as np

def read_data(data_folder):
    train = pd.read_csv(data_folder + '/train.csv')
    test = pd.read_csv(data_folder + '/test.csv')
    return train, test

#anatom_siteのone_hot_encoder
def anatom_site(train, test):
    concat = pd.concat([train['anatom_site_general_challenge'], test['anatom_site_general_challenge']], ignore_index = True)
    dummies = pd.get_dummies(concat, dummy_na = True, dtype = np.uint8, prefix = 'site')
    train = pd.concat([train, dummies.iloc[:train.shape[0]]], axis = 1)
    test = pd.concat([test, dummies.iloc[train.shape[0]:].reset_index(drop = True)], axis = 1)

    return train, test

def preprocessing(train, test):
    #sexのmapping
    train['sex'] = train['sex'].map({'male' : 1, 'female' : 0})
    test['sex'] = test['sex'].map({'male' : 1, 'female' : 0})
    train['sex'] = train['sex'].fillna(-1)
    test['sex'] = test['sex'].fillna(-1)

    #age_approxの正規化
    train['age_approx'] /= train['age_approx'].max()
    test['age_approx'] /= test['age_approx'].max()
    train['age_approx'] = train['age_approx'].fillna(0)
    test['age_approx'] = test['age_approx'].fillna(0)

    #その他
    train['patient_id'] = train['patient_id'].fillna(0)
    train['fold'] = train['tfrecord']

    return train, test

def create_data(data_folder):
    train, test = read_data(data_folder)
    train, test = anatom_site(train, test)
    train, test = preprocessing(train, test)

    return train, test