import efficientnet.tfkeras as efn
from keras.layers import GlobalAveragePooling2D, Dense, Activation
from keras.models import Model

"""
モデルを作成する関数
"""

def efficientnet():
    first = efn.EfficientNetB7(input_shape = (224, 224, 3), weights = 'imagenet', include_top = False)
    x = first.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1)(x)
    output = Activation('sigmoid')(x)
    model = Model(first.input, output)

    return model