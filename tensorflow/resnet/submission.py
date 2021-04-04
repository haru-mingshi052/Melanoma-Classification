import numpy as np
import cv2

from pre_submission import create_df, read_sub
from train import train

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)

import logging
tf.get_logger().setLevel(logging.ERROR)

import argparse

"""
submissionファイルを作成する関数
"""

parser = argparse.ArgumentParser(
    description = "folder path"
)

parser.add_argument("--data_folder", default = '/kaggle/input/siim-isic-melanoma-classification', type = str,
                    help = "データの入っているフォルダ")
parser.add_argument('--output_folder', default = '/kaggle/working', type = str,
                    help = "提出用ファイルを出力するフォルダ")

args = parser.parse_args()

#====================
# submission
#====================
def submission():
    model = train(args.data_folder) #学習

    df_test = create_df(args.data_folder) #テストファイル読み込み
    sub = read_sub(args.data_folder) #submissionファイル読み込み

    #推論
    target = []
    for path in df_test['images']:
        img = cv2.imread(str(path))
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        img = np.reshape(img, (1, 224, 224, 3))
        prediction = model.predict(img)
        target.append(prediction[0][0])

    sub['target'] = target

    sub.to_csv(args.output_folder + '/submission.csv', index = False)

if __name__ == '__main__':
    submission()