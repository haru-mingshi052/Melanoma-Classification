import tensorflow
from keras.layers import Conv2D, MaxPooling2D, Activation, GlobalAveragePooling2D, Dense, Input
from keras.optimizers import Adam
from keras.models import Model

"""
モデルを作成する関数
"""

#=====================
# create_model
#=====================
def create_model():
    # input
    input_tensor = Input(shape = (224, 224, 3))
    # 1
    Conv1_1 = Conv2D(64, (3,3), padding= 'same')(input_tensor)
    activation1_1 = Activation('relu')(Conv1_1)
    Conv1_2 = Conv2D(64, (3,3), padding = 'same')(activation1_1)
    activation1_2 = Activation('relu')(Conv1_2)
    MP_1 = MaxPooling2D((2,2))(activation1_2)
    # 2
    Conv2_1 = Conv2D(128, (3,3), padding = 'same')(MP_1)
    activation2_1 = Activation('relu')(Conv2_1)
    Conv2_2 = Conv2D(128, (3,3), padding = 'same')(activation2_1)
    activation2_2 = Activation('relu')(Conv2_2)
    MP_2 = MaxPooling2D((2,2))(activation2_2)
    # 3
    Conv3_1 = Conv2D(256, (3,3), padding = 'same')(MP_2)
    activation3_1 = Activation('relu')(Conv3_1)
    Conv3_2 = Conv2D(256, (3,3), padding = 'same')(activation3_1)
    activation3_2 = Activation('relu')(Conv3_2)
    Conv3_3 = Conv2D(256, (3,3), padding = 'same')(activation3_2)
    activation3_3 = Activation('relu')(Conv3_3)
    MP_3 = MaxPooling2D((2,2))(activation3_3)
    # 4
    Conv4_1 = Conv2D(512, (3,3), padding = 'same')(MP_3)
    activation4_1 = Activation('relu')(Conv4_1)
    Conv4_2 = Conv2D(512, (3,3), padding = 'same')(activation4_1)
    activation4_2 = Activation('relu')(Conv4_2)
    Conv4_3 = Conv2D(512, (3,3), padding = 'same')(activation4_2)
    activation4_3 = Activation('relu')(Conv4_3)
    MP_4 = MaxPooling2D((2,2))(activation4_3)
    # 5
    Conv5_1 = Conv2D(512, (3,3), padding = 'same')(MP_4)
    activation5_1 = Activation('relu')(Conv5_1)
    Conv5_2 = Conv2D(512, (3,3), padding = 'same')(activation5_1)
    activation5_2 = Activation('relu')(Conv5_2)
    Conv5_3 = Conv2D(512, (3,3), padding = 'same')(activation5_2)
    activation5_3 = Activation('relu')(Conv5_3)
    MP_5 = MaxPooling2D((2,2))(activation5_3)
    # output
    GP = GlobalAveragePooling2D()(MP_5)
    dense = Dense(1)(GP)
    output_tensor = Activation('sigmoid')(dense)

    model = Model(input_tensor, output_tensor)

    model.compile(
        optimizer = Adam(lr = 1e-5),
        loss = tensorflow.keras.losses.binary_crossentropy,
        metrics = [tensorflow.keras.metrics.AUC()]
    )

    return model