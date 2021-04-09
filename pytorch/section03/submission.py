if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd

from train import train
from seed_everything import seed_everything

import argparse

"""
submissionファイルを作成する関数
"""

parser = argparse.ArgumentParser(
    description = "parameter for training"
)

parser.add_argument("--data_folder", default = '/kaggle/input/jpeg-melanoma-256x256', type = str,
                    help = 'データの入っているフォルダ')
parser.add_argument('--output_folder', default = '/kaggle/working', type = str,
                    help = "提出用ファイルを出力するフォルダ")
parser.add_argument('--es_patience', default = 3, type = int,
                    help = "どれだけ改善が無い場合学習を止めるか")
parser.add_argument('--epochs', default = 15, type = int,
                    help = "何エポック学習するか")
parser.add_argument('--TTA', default = 3, type = int,
                    help = 'TTAの値')
parser.add_argument('--model_name', default = 'pytorch-1', type = str, choices = ['pytorch-1', 'pytorch-2', 'pytorch-3'],
                    help = '[pytorch-1] or [pytorch-2] or [pytorch-3]')

args = parser.parse_args()

def submission():

    seed_everything(71)

    preds, oof = train(
        data_folder = args.data_folder,
        output_folder = args.output_folder,
        es_patience = args.es_patience,
        epochs = args.epochs,
        TTA = args.TTA,
        model_name = args.model_name,
    )

    pd.Series(oof.reshape(-1)).to_csv(args.data_folder + '/oof.csv', index = False)

    sub = pd.read_csv(args.data_folder + '/sample_submission.csv')
    sub['target'] = preds.cpu().numpy().reshape(-1,)
    sub.to_csv(args.output_folder + '/submission.csv', index = False)

if __name__ == '__main__':
    submission()