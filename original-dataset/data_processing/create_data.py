from preprocessing import *
from image_processing import resize_image
import argparse

"""
データを加工する関数を集めた関数
"""

parser = argparse.ArgumentParser(
    description = "data augmentation"
)

parser.add_argument("--data_folder", default = "/kaggle/input/siim-isic-melanoma-classification", type = str,
                    help = "データのあるフォルダー")
parser.add_argument("--output_folder", default = '/kaggle/working', type = str,
                    help = "加工したデータを出力したいフォルダー")

args = parser.parse_args()

#============================
# melanoma_dataset
#============================
def melanoma_dataset():
    train, test, sub = read_data(args.data_folder)
    train = add_fold(train)
    train, test = anatom_site(train, test)
    train, test = sex_onehot(train, test)
    train, test = age_group(train, test)
    train, test = age_group_onehot(train, test)
    train, test = age_approx(train, test)
    train, test = delete_column(train, test)
    
    train.to_csv(args.output_folder + '/train.csv', index = False)
    test.to_csv(args.output_folder + '/test.csv', index = False)
    sub.to_csv(args.output_folder + '/sample_submission.csv', index = False)

    resize_image(args.data_folder, args.output_folder)

if __name__ == '__main__':
    melanoma_dataset()