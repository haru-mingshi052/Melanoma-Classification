import tensorflow

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(71)

"""
学習の準備をする関数
    read_data：学習データの準備をする関数
    create_df：generator用のデータフレームを用意する関数
    split：データを分割する関数
"""

#==========================
# read_data
#==========================
def read_data(data_folder):
    train = pd.read_csv(data_folder + '/train.csv')

    #データ量が多いのでアンダーサンプリング
    df_0 = train[train['target'] == 0].sample(2000)
    df_1 = train[train['target'] == 1]
    train = pd.concat([df_0, df_1])
    train = train.reset_index()

    return train

#==========================
# create_df
#==========================
def create_df(data_folder):
    train_dir = data_folder + '/jpeg/train/'
    train = read_data(data_folder)

    #image_nameを絶対パスへ
    labels = []
    data = []
    for i in range(train.shape[0]):
        data.append(train_dir + train['image_name'].iloc[i] + '.jpg')
        labels.append(train['target'].iloc[i])
    
    df = pd.DataFrame(data)
    df.columns = ['images']
    df['target'] = labels

    return split(df)

#===========================
# split
#===========================
def split(df):
    return train_test_split(df, test_size =0.2, random_state = 71)