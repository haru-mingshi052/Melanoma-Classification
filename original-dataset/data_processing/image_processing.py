import os
import glob
import shutil
from PIL import Image

"""
画像を加工する関数
"""

#===========================
# resize_image
#===========================
def resize_image(data_folder, output_folder):
    train_files = glob.glob(data_folder + '/jpeg/train/*.jpg')
    test_files = glob.glob(data_folder + '/jpeg/test/*.jpg')

    train_dir = output_folder + '/train'
    test_dir = output_folder + '/test'

    os.mkdir(train_dir)
    os.mkdir(test_dir)

    for f in train_files:
        img = Image.open(f)
        img_resize = img.resize((256, 256))
        title = f[-16:]
        img_resize.save(train_dir + '/' + title)

    for f in test_files:
        img = Image.open(f)
        img_resize = img.resize((256, 256))
        title = f[-16:]
        img_resize.save(test_dir + '/' + title)

    shutil.make_archive(output_folder + '/train', 'zip', root_dir = train_dir)
    shutil.make_archive(output_folder + '/test', 'zip', root_dir = test_dir)

    shutil.rmtree(train_dir)
    shutil.rmtree(test_dir)
