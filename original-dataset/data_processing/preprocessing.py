import pandas as pd
import numpy as np

"""
データを加工する関数
"""

#=============================
# read_data
#=============================
def read_data(data_folder):
    train = pd.read_csv(data_folder + '/train.csv')
    test = pd.read_csv(data_folder + '/test.csv')
    sub = pd.read_csv(data_folder + '/sample_submission.csv')

    return train, test, sub

"""
やりたい加工の整理
～追加編～
・age_groupの作成 → one_hot_enocder
・anatom_site_general_challenge（作成済み）→ one_hot_encoder
・sex（作成済み）→ one_hot_encoder
・foldの追加
～削除編～
・age_group
・anatom_site_general_challenge
・sex
・patient_id
・diagnosis
・benign-malignant
～正規化編～
・age_approxの正規化
"""

#==============================
# add_fold
# foldの追加
#==============================
def add_fold(train):
    """
    被っているデータは'fold' = -1
    target == 1のデータ：'fold' = 0 ~ 14
    target == 0のデータ：'fold' = 0 ~ 14
    """
    # 被っているデータ　主催者から提供されている
    duplicate = pd.read_csv('/kaggle/input/siim2020duplicatedataset/2020_Challenge_duplicates.csv')

    # duplicateの中から'train'の分だけ抜き出す
    duplicate = duplicate[duplicate['partition'] == 'train'].reset_index(drop = True)

    # 被っているデータのidをリストへ
    duplicate_list = []
    for i in range(len(duplicate)):
        duplicate_list.append(duplicate.iloc[i,1])

    # fold用のDataFrameを作成
    df_fold = train.loc[:,['image_name', 'target']]
    df_fold['fold'] = np.nan

    # 被っているデータのfoldを-1に
    for i in range(len(df_fold)):
        if df_fold.iat[i,0] in duplicate_list:
            df_fold.iat[i,2] = -1

    # 今回は陽性データが少ないため、１個の’fold’に陽性データが偏らないようにデータを分けて’fold’の番号を振っていく
    # 被っているデータ
    df_duplicate = df_fold[df_fold['fold'] == -1].reset_index(drop = True)
    # 陰性かつ被っていないデータ
    df_benign = df_fold[(df_fold['target'] == 0) & (df_fold['fold'] != -1)].reset_index(drop = True)
    # 陽性かつ被っていないデータ
    df_malign = df_fold[(df_fold['target'] == 1) & (df_fold['fold'] != -1)].reset_index(drop = True)

    # データをシャッフル
    df_benign = df_benign.sample(frac = 1, random_state = 47).reset_index(drop = True)
    df_malign = df_malign.sample(frac = 1, random_state = 47).reset_index(drop = True)

    # benignのfoldに番号を振る
    for i in range(len(df_benign)):
        df_benign.iat[i,2] = i % 15

    # malignのfoldに番号を振る
    for i in range(len(df_malign)):
        df_malign.iat[i,2] = i % 15

    df_fold = pd.concat([df_duplicate, df_benign], axis = 0)
    df_fold = pd.concat([df_fold, df_malign], axis = 0)

    # 作成した’fold’を含むデータとtrainデータを’image_name’をキーにして結合
    train = pd.merge(train, df_fold, on = ['image_name', 'target'])

    return train

#=========================================
# anatom_site
# anatom_siteのone_hot_encoder
#=========================================
def anatom_site(train, test):
    concat = pd.concat([train['anatom_site_general_challenge'], test['anatom_site_general_challenge']], ignore_index = True)
    dummies = pd.get_dummies(concat, dummy_na = True, dtype = np.uint8, prefix = 'site')
    train = pd.concat([train, dummies.iloc[:train.shape[0]]], axis = 1)
    test = pd.concat([test, dummies.iloc[train.shape[0]:].reset_index(drop = True)], axis = 1)
    return train, test

#=========================================
# sex_onehot
# sexのone_hot_encoder
#=========================================
def sex_onehot(train, test):
    concat = pd.concat([train['sex'], test['sex']], ignore_index = True)
    dummies = pd.get_dummies(concat, dummy_na = True, dtype = np.uint8, prefix = 'sex')
    train = pd.concat([train, dummies.iloc[:train.shape[0]]], axis = 1)
    test = pd.concat([test, dummies.iloc[train.shape[0]:].reset_index(drop = True)], axis = 1)
    return train, test

#=======================================
# age_group
# age_groupの作成
#=======================================
def age_group(train, test):
    teen = [0, 5, 10, 15, 20]
    young_adult = [25, 30, 35, 40]
    middle_adult = [45, 50, 55]
    young_old = [60, 65, 70]
    middle_old = [75, 80, 90]

    train['age_group'] = 'empty'
    test['age_group'] = 'empty'

    for i in range(len(train)):
        if train.iat[i, 3] in teen:
            train.iat[i, 19] = 'teen'
        elif train.iat[i, 3] in young_adult:
            train.iat[i, 19] = 'young_adult'
        elif train.iat[i, 3] in middle_adult:
            train.iat[i, 19] = 'middle_adult'
        elif train.iat[i, 3] in young_old:
            train.iat[i, 19] = 'young_old'
        elif train.iat[i, 3] in middle_old:
            train.iat[i, 19] = 'middle_old'

    for i in range(len(test)):
        if test.iat[i, 3] in teen:
            test.iat[i, 15] = 'teen'
        elif test.iat[i, 3] in young_adult:
            test.iat[i, 15] = 'young_adult'
        elif test.iat[i, 3] in middle_adult:
            test.iat[i, 15] = 'middle_adult'
        elif test.iat[i, 3] in young_old:
            test.iat[i, 15] = 'young_old'
        elif test.iat[i, 3] in middle_old:
            test.iat[i, 15] = 'middle_old'

    return train, test

#============================================
# age_group_onehot
# age_groupのone_hot_encoder
#============================================
def age_group_onehot(train, test):
    concat = pd.concat([train['age_group'], test['age_group']], ignore_index = True)
    dummies = pd.get_dummies(concat, dummy_na = True, dtype = np.uint8, prefix = 'age')
    train = pd.concat([train, dummies.iloc[:train.shape[0]]], axis = 1)
    test = pd.concat([test, dummies.iloc[train.shape[0]:].reset_index(drop = True)], axis = 1)

    return train, test

#===========================================
# age_approx
# age_approxの正規化
#===========================================
def age_approx(train, test):
    train['age_approx'] /= train['age_approx'].max()
    test['age_approx'] /= test['age_approx'].max()
    train['age_approx'] = train['age_approx'].fillna(0)
    test['age_approx'] = test['age_approx'].fillna(0)

    return train, test

#============================================
# delete_column
# データの削除
#============================================
def delete_column(train, test):
    train_drop_list = ['patient_id', 'sex', 'anatom_site_general_challenge', 'diagnosis', 'benign_malignant', 'age_group', 'age_nan']
    test_drop_list = ['patient_id', 'sex', 'anatom_site_general_challenge', 'age_group', 'age_nan']
    train.drop(train_drop_list, axis = 1, inplace = True)
    test.drop(test_drop_list, axis = 1, inplace = True)

    return train, test