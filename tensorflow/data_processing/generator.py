from keras.preprocessing.image import ImageDataGenerator

from .processing import create_df

"""
学習する用のgeneratorを作成する関数
"""
#=======================
# create_generator
#=======================
def create_generator(data_folder):
    train_datagen = ImageDataGenerator(
        rescale = 1./255, #normalizing
        rotation_range = 20, #回転範囲
        width_shift_range = 0.2, #平行移動させる範囲
        height_shift_range = 0.2, #平行移動させる範囲
        horizontal_flip = True #水平方向にランダムに反転
    )

    val_datagen = ImageDataGenerator(
        rescale = 1./225
    )

    train, val = create_df(data_folder)

    train_generator = train_datagen.flow_from_dataframe(
        train,
        x_col = 'images',
        y_col = 'target',
        target_size = (224, 224),
        batch_size = 8,
        shuffle = True,
        class_mode = 'raw'
    )

    val_generator = val_datagen.flow_from_dataframe(
        val,
        x_col = 'images',
        y_col = 'target',
        target_size = (224, 224),
        batch_size = 8,
        shuffle = False,
        class_mode = 'raw'
    )

    return train_generator, val_generator