import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dropout, GlobalAveragePooling2D, MaxPooling2D, add, Dense, Reshape, multiply
from keras.models import Model
from tensorflow.keras.utils import get_custom_objects

"""
モデルを作成する関数
"""

class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))

get_custom_objects().update({'Mish': Mish(mish)})

def senet():
    input = Input(shape = (224,224,3))

    # 1層目
    conv1_1 = Conv2D(64, (3, 3), padding = 'same')(input)
    bn1_1 = BatchNormalization()(conv1_1)
    activation1_1 = Activation('Mish')(bn1_1)
    drop1_1 = Dropout(0.5)(activation1_1)
    conv1_2 = Conv2D(64, (3,3), padding = 'same')(drop1_1)
    bn1_2 = BatchNormalization()(conv1_2)
    activation1_2 = Activation('Mish')(bn1_2)
    drop1_2 = Dropout(0.5)(activation1_2)
    conv1_3 = Conv2D(64, (3,3), padding = 'same')(drop1_2)
    bn1_3 = BatchNormalization()(conv1_3)
    activation1_3 = Activation('Mish')(bn1_3)
    SEgap1 = GlobalAveragePooling2D()(activation1_3)
    SEresh1 = Reshape((1,1,64))(SEgap1)
    SEconv1_1 = Conv2D(16, (1,1))(SEresh1)
    SEact1_1 = Activation('sigmoid')(SEconv1_1)
    SEconv1_2 = Conv2D(64, (1,1))(SEact1_1)
    SEact1_2 = Activation('sigmoid')(SEconv1_2)
    mul1 = multiply([activation1_3, SEact1_2])
    add1 = add([mul1, activation1_1])
    mp1 = MaxPooling2D((2,2))(add1)

    # 2層目
    conv2_1 = Conv2D(128, (3, 3), padding = 'same')(mp1)
    bn2_1 = BatchNormalization()(conv2_1)
    activation2_1 = Activation('Mish')(bn2_1)
    drop2_1 = Dropout(0.5)(activation2_1)
    conv2_2 = Conv2D(128, (3,3), padding = 'same')(drop2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    activation2_2 = Activation('Mish')(bn2_2)
    drop2_2 = Dropout(0.5)(activation2_2)
    conv2_3 = Conv2D(128, (3,3), padding = 'same')(drop2_2)
    bn2_3 = BatchNormalization()(conv2_3)
    activation2_3 = Activation('Mish')(bn2_3)
    SEgap2 = GlobalAveragePooling2D()(activation2_3)
    SEresh2 = Reshape((1,1,128))(SEgap2)
    SEconv2_1 = Conv2D(32, (1,1))(SEresh2)
    SEact2_1 = Activation('sigmoid')(SEconv2_1)
    SEconv2_2 = Conv2D(128, (1,1))(SEact2_1)
    SEact2_2 = Activation('sigmoid')(SEconv2_2)
    mul2 = multiply([activation2_3, SEact2_2])
    add2 = add([mul2, activation2_1])
    mp2 = MaxPooling2D((2,2))(add2)

    # 3層目
    conv3_1 = Conv2D(256, (3, 3), padding = 'same')(mp2)
    bn3_1 = BatchNormalization()(conv3_1)
    activation3_1 = Activation('Mish')(bn3_1)
    drop3_1 = Dropout(0.5)(activation3_1)
    conv3_2 = Conv2D(256, (3,3), padding = 'same')(drop3_1)
    bn3_2 = BatchNormalization()(conv3_2)
    activation3_2 = Activation('Mish')(bn3_2)
    drop3_2 = Dropout(0.5)(activation3_2)
    conv3_3 = Conv2D(256, (3,3), padding = 'same')(drop3_2)
    bn3_3 = BatchNormalization()(conv3_3)
    activation3_3 = Activation('Mish')(bn3_3)
    SEgap3 = GlobalAveragePooling2D()(activation3_3)
    SEresh3 = Reshape((1,1,256))(SEgap3)
    SEconv3_1 = Conv2D(32, (1,1))(SEresh3)
    SEact3_1 = Activation('sigmoid')(SEconv3_1)
    SEconv3_2 = Conv2D(256, (1,1))(SEact3_1)
    SEact3_2 = Activation('sigmoid')(SEconv3_2)
    mul3 = multiply([activation3_3, SEact3_2])
    add3 = add([mul3, activation3_1])
    mp3 = MaxPooling2D((2,2))(add3)

    # 4層目
    conv4_1 = Conv2D(512, (3, 3), padding = 'same')(mp3)
    bn4_1 = BatchNormalization()(conv4_1)
    activation4_1 = Activation('Mish')(bn4_1)
    drop4_1 = Dropout(0.5)(activation4_1)
    conv4_2 = Conv2D(512, (3,3), padding = 'same')(drop4_1)
    bn4_2 = BatchNormalization()(conv4_2)
    activation4_2 = Activation('Mish')(bn4_2)
    drop4_2 = Dropout(0.5)(activation4_2)
    conv4_3 = Conv2D(512, (3,3), padding = 'same')(drop4_2)
    bn4_3 = BatchNormalization()(conv4_3)
    activation4_3 = Activation('Mish')(bn4_3)
    SEgap4 = GlobalAveragePooling2D()(activation4_3)
    SEresh4 = Reshape((1,1,512))(SEgap4)
    SEconv4_1 = Conv2D(32, (1,1))(SEresh4)
    SEact4_1 = Activation('sigmoid')(SEconv4_1)
    SEconv4_2 = Conv2D(512, (1,1))(SEact4_1)
    SEact4_2 = Activation('sigmoid')(SEconv4_2)
    mul4 = multiply([activation4_3, SEact4_2])
    add4 = add([mul4, activation4_1])
    mp4 = MaxPooling2D((2,2))(add4)

    # 5層目
    conv5_1 = Conv2D(512, (3, 3), padding = 'same')(mp4)
    bn5_1 = BatchNormalization()(conv5_1)
    activation5_1 = Activation('relu')(bn5_1)
    drop5_1 = Dropout(0.5)(activation5_1)
    conv5_2 = Conv2D(512, (3,3), padding = 'same')(drop5_1)
    bn5_2 = BatchNormalization()(conv5_2)
    activation5_2 = Activation('relu')(bn5_2)
    drop5_2 = Dropout(0.5)(activation5_2)
    conv5_3 = Conv2D(512, (3,3), padding = 'same')(drop5_2)
    bn5_3 = BatchNormalization()(conv5_3)
    activation5_3 = Activation('relu')(bn5_3)
    SEgap5 = GlobalAveragePooling2D()(activation5_3)
    SEresh5 = Reshape((1,1,512))(SEgap5)
    SEconv5_1 = Conv2D(32, (1,1))(SEresh5)
    SEact5_1 = Activation('sigmoid')(SEconv5_1)
    SEconv5_2 = Conv2D(512, (1,1))(SEact5_1)
    SEact5_2 = Activation('sigmoid')(SEconv5_2)
    mul5 = multiply([activation5_3, SEact5_2])
    add5 = add([mul5, activation5_1])
    mp5 = MaxPooling2D((2,2))(add5)

    # 6層目
    conv6_1 = Conv2D(512, (3, 3), padding = 'same')(mp5)
    bn6_1 = BatchNormalization()(conv6_1)
    activation6_1 = Activation('relu')(bn6_1)
    drop6_1 = Dropout(0.5)(activation6_1)
    conv6_2 = Conv2D(512, (3,3), padding = 'same')(drop6_1)
    bn6_2 = BatchNormalization()(conv6_2)
    activation6_2 = Activation('relu')(bn6_2)
    drop6_2 = Dropout(0.5)(activation6_2)
    conv6_3 = Conv2D(512, (3,3), padding = 'same')(drop6_2)
    bn6_3 = BatchNormalization()(conv6_3)
    activation6_3 = Activation('relu')(bn6_3)
    SEgap6 = GlobalAveragePooling2D()(activation6_3)
    SEresh6 = Reshape((1,1,512))(SEgap6)
    SEconv6_1 = Conv2D(32, (1,1))(SEresh6)
    SEact6_1 = Activation('sigmoid')(SEconv6_1)
    SEconv6_2 = Conv2D(512, (1,1))(SEact6_1)
    SEact6_2 = Activation('sigmoid')(SEconv6_2)
    mul6 = multiply([activation6_3, SEact6_2])
    add6 = add([mul6, activation6_1])
    mp6 = MaxPooling2D((2,2))(add6)

    # 7層目
    conv7_1 = Conv2D(512, (3, 3), padding = 'same')(mp6)
    bn7_1 = BatchNormalization()(conv7_1)
    activation7_1 = Activation('relu')(bn7_1)
    drop7_1 = Dropout(0.5)(activation7_1)
    conv7_2 = Conv2D(512, (3,3), padding = 'same')(drop7_1)
    bn7_2 = BatchNormalization()(conv7_2)
    activation7_2 = Activation('relu')(bn7_2)
    drop7_2 = Dropout(0.5)(activation7_2)
    conv7_3 = Conv2D(512, (3,3), padding = 'same')(drop7_2)
    bn7_3 = BatchNormalization()(conv7_3)
    activation7_3 = Activation('relu')(bn7_3)
    SEgap7 = GlobalAveragePooling2D()(activation7_3)
    SEresh7 = Reshape((1,1,512))(SEgap7)
    SEconv7_1 = Conv2D(32, (1,1))(SEresh7)
    SEact7_1 = Activation('sigmoid')(SEconv7_1)
    SEconv7_2 = Conv2D(512, (1,1))(SEact7_1)
    SEact7_2 = Activation('sigmoid')(SEconv7_2)
    mul7 = multiply([activation7_3, SEact7_2])
    add7 = add([mul7, activation7_1])
    mp7 = MaxPooling2D((2,2))(add7)

    # output
    GP = GlobalAveragePooling2D()(mp7)
    dense = Dense(1)(GP)
    output = Activation('sigmoid')(dense)

    model = Model(input, output)

    return model