import pandas as pd

"""
submissionの準備をする関数
    read_data：testファイルを読み込む関数
    read_sub：submissionファイルを読み込む関数
    create_df：generatorを作成する用のデータフレームを作成する関数
"""

#=========================
# read_data
#=========================
def read_data(data_folder):
    test = pd.read_csv(data_folder + '/test.csv')
    return test

#=========================
# read_sub
#=========================
def read_sub(data_folder):
    sub = pd.read_csv(data_folder + '/sample_submission.csv')
    return sub

#=========================
# create_df
#=========================
def create_df(data_folder):
    test_dir = data_folder + '/jpeg/test/'
    test = read_data(data_folder)

    #image_nameを絶対パスへ
    test_data = []
    for i in range(test.shape[0]):
        test_data.append(test_dir + test['image_name'].iloc[i] + '.jpg')
    df_test = pd.DataFrame(test_data)
    df_test.columns = ['images']

    return df_test