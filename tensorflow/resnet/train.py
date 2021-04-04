from model import create_model
from generator import create_generator

"""
train：モデルを学習させる関数
"""

#=======================
# train
#=======================
def train(data_folder):
    train_generator, val_generator = create_generator(data_folder)
    model = create_model()

    model.fit_generator(
        train_generator,
        steps_per_epoch = 100,
        epochs = 10,
        validation_data = val_generator,
        validation_steps = 50
    )

    return model