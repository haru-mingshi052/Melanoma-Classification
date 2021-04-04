import tensorflow
from keras.layers import Input, Conv2D, BatchNormalization, Activation, GlobalAveragePooling2D, MaxPooling2D, add, Dense
from keras.optimizers import Adam
from keras.models import Model

"""
モデルを作成する関数
"""

#=====================
# create_model
#=====================
def create_model():
    input_tensor = Input(shape = (224, 224, 3))

    # 1
    conv1_1 = Conv2D(64, (3, 3), padding = 'same')(input_tensor)
    bn1_1 = BatchNormalization()(conv1_1)
    activation1_1 = Activation('relu')(bn1_1)
    conv1_2 = Conv2D(64, (3,3), padding = 'same')(activation1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    activation1_2 = Activation('relu')(bn1_2)
    conv1_3 = Conv2D(64, (3,3), padding = 'same')(activation1_2)
    bn1_3 = BatchNormalization()(conv1_3)
    activation1_3 = Activation('relu')(bn1_3)
    add1 = add([activation1_3, activation1_1])
    mp1 = MaxPooling2D((2,2))(add1)

    # 2
    conv2_1 = Conv2D(128, (3,3), padding = 'same')(mp1)
    bn2_1 = BatchNormalization()(conv2_1)
    activation2_1 = Activation('relu')(bn2_1)
    conv2_2 = Conv2D(128, (3,3), padding = 'same')(activation2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    activation2_2 = Activation('relu')(bn2_2)
    conv2_3 = Conv2D(128, (3,3), padding = 'same')(activation2_2)
    bn2_3 = BatchNormalization()(conv2_3)
    activation2_3 = Activation('relu')(bn2_3)
    add2 = add([activation2_3, activation2_1])
    mp2 = MaxPooling2D((2,2))(add2)

    # 3
    conv3_1 = Conv2D(256, (3,3), padding = 'same')(mp2)
    bn3_1 = BatchNormalization()(conv3_1)
    activation3_1 = Activation('relu')(bn3_1)
    conv3_2 = Conv2D(256, (3,3), padding = 'same')(activation3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    activation3_2 = Activation('relu')(bn3_2)
    conv3_3 = Conv2D(256, (3,3), padding = 'same')(activation3_2)
    bn3_3 = BatchNormalization()(conv3_3)
    activation3_3 = Activation('relu')(bn3_3)
    add3 = add([activation3_3, activation3_1])
    mp3 = MaxPooling2D((2,2))(add3)

    # 4
    conv4_1 = Conv2D(512, (3,3), padding = 'same')(mp3)
    bn4_1 = BatchNormalization()(conv4_1)
    activation4_1 = Activation('relu')(bn4_1)
    conv4_2 = Conv2D(512, (3,3), padding = 'same')(activation4_1)
    bn4_2 = BatchNormalization()(conv4_2)
    activation4_2 = Activation('relu')(bn4_2)
    conv4_3 = Conv2D(512, (3,3), padding = 'same')(activation4_2)
    bn4_3 = BatchNormalization()(conv4_3)
    activation4_3 = Activation('relu')(bn4_3)
    add4 = add([activation4_3, activation4_1])
    mp4 = MaxPooling2D((2,2))(add4)

    # 5
    conv5_1 = Conv2D(512, (3,3), padding = 'same')(mp4)
    bn5_1 = BatchNormalization()(conv5_1)
    activation5_1 = Activation('relu')(bn5_1)
    conv5_2 = Conv2D(512, (3,3), padding = 'same')(activation5_1)
    bn5_2 = BatchNormalization()(conv5_2)
    activation5_2 = Activation('relu')(bn5_2)
    conv5_3 = Conv2D(512, (3,3), padding = 'same')(activation5_2)
    bn5_3 = BatchNormalization()(conv5_3)
    activation5_3 = Activation('relu')(bn5_3)
    add5 = add([activation5_3, activation5_1])
    mp5 = MaxPooling2D((2,2))(add5)

    # output
    GP = GlobalAveragePooling2D()(mp5)
    dense = Dense(1)(GP)
    output_tensor = Activation('sigmoid')(dense)

    model = Model(input_tensor, output_tensor)

    model.compile(
        optimizer = Adam(lr = 1e-5),
        loss = tensorflow.keras.losses.binary_crossentropy,
        metrics = [tensorflow.keras.metrics.AUC()]
    )

    return model