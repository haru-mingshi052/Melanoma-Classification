import tensorflow
from keras.optimizers import Adam
from Models import *
from data_processing import create_generator

"""
train：モデルを学習させる関数
"""

#=======================
# train
#=======================
def train(data_folder, model_name):
    train_generator, val_generator = create_generator(data_folder)

    if model_name == 'VGG16':
        model = vgg16()
    elif model_name == "ResNet":
        model = resnet()
    elif model_name == "SENet":
        model = senet()
    elif model_name == 'EfficientNet':
        model = efficientnet()

    model.compile(
        optimizer = Adam(lr = 1e-5),
        loss = tensorflow.keras.losses.binary_crossentropy,
        metrics = [tensorflow.keras.metrics.AUC()]
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 10,
        validation_data = val_generator,
        validation_steps = 50
    )

    return model