import tensorflow.keras as keras
import tensorflow as tf
from keras import layers

class ResBlock(keras.Model):
    def __init__(self, channels, strides=1):
        super().__init__()

        self.conv1 = layers.Conv2D(channels, 3, strides=strides, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(channels, 3, strides=1, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()

        if strides != 1:
            self.shortcut = keras.Sequential([
                layers.Conv2D(channels, 1, strides, padding='same', use_bias=False)
            ])
        else:
            self.shortcut = keras.Sequential()
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        shortcut = self.shortcut(inputs)
        return tf.nn.relu(tf.add(x, shortcut))
    
    def get_config(self):
        return {}

class Encoder(keras.Model):
    def __init__(self, channels, repeat, strides):
        super().__init__()
        self.resBlocks = keras.Sequential()
        self.resBlocks.add(ResBlock(channels, strides))
        for _ in range(1, repeat):
            self.resBlocks.add(ResBlock(channels, strides=1))

    def call(self, inputs):
        return self.resBlocks(inputs)
    
    def get_config(self):
        return {}

class ChannelAttention(keras.Model):
    def __init__(self, reduction):
        super().__init__()
        self.globalMaxPool = layers.GlobalMaxPooling2D(keepdims=True)
        self.globalAvgPool = layers.GlobalAveragePooling2D(keepdims=True)
        self.reduction = reduction

    def build(self, input_shape):
        self.fc = keras.Sequential([
            layers.Conv2D(input_shape[-1]//self.reduction, 3, padding='same'),
            layers.ReLU(),
            layers.Conv2D(input_shape[-1], 1, padding='valid')
        ])

    def call(self, inputs):
        x1 = self.globalMaxPool(inputs)
        x2 = self.globalAvgPool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x = tf.nn.sigmoid(layers.add([x1, x2]))
        return x


class SpatialAttention(keras.Model):
    def __init__(self):
        super().__init__()
        self.conv3x3 = layers.Conv2D(1, 3, padding='same')

    def call(self, inputs):
        # https://github.com/kobiso/CBAM-tensorflow/blob/master/attention_module.py#L95
        x1 = tf.math.reduce_max(inputs, axis=3, keepdims=True)
        x2 = tf.math.reduce_mean(inputs, axis=3, keepdims=True)
        x = tf.concat([x1, x2], 3)
        x = self.conv3x3(x)
        x = tf.nn.sigmoid(x)
        return x


class CBAM(keras.Model):
    def __init__(self, reduction):
        super().__init__()
        self.channelAttention = ChannelAttention(reduction)
        self.spaialAttention = SpatialAttention()

    def call(self, inputs):
        x = inputs * self.channelAttention(inputs)
        x = x * self.spaialAttention(x)
        return x


class Decoder(keras.Model):
    def __init__(self, channels, upsample=True):
        super().__init__()
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

        if upsample:
            self.upsample = keras.Sequential([
                layers.UpSampling2D(2, interpolation='nearest')
            ])
        else:
            self.upsample = keras.Sequential()

        self.conv3x3_2 = layers.Conv2D(
            channels, 3, padding='same', use_bias=False)
        self.conv1x1 = layers.Conv2D(channels, 1, use_bias=False)

    def build(self, input_shape):
        self.conv3x3_1 = layers.Conv2D(
            input_shape[-1], 3, padding='same', use_bias=False)

    def call(self, inputs):
        x = self.bn1(inputs)
        x = tf.nn.relu(x)
        x = self.upsample(x)
        x = self.conv3x3_1(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)
        x = self.conv3x3_2(x)

        shortcut = self.conv1x1(self.upsample(inputs))
        x += shortcut
        return x

    def get_config(self):
        return {}


def ResNet34UNetV2(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Encode by ResNet34
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x0 = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # ResNet34
    x1 = Encoder(64, 3, strides=1)(CBAM(reduction=16)(x0))
    x2 = Encoder(128, 4, strides=2)(CBAM(reduction=16)(x1))
    x3 = Encoder(256, 6, strides=2)(CBAM(reduction=16)(x2))
    x4 = Encoder(512, 3, strides=2)(CBAM(reduction=16)(x3))

    # Center Block
    y5 = layers.Conv2D(512, 3, padding='same', use_bias=False)(x4)

    # Decode
    y4 = Decoder(64)(layers.Concatenate(axis=3)([x4, y5]))
    y3 = Decoder(64)(layers.Concatenate(axis=3)([x3, y4]))
    y2 = Decoder(64)(layers.Concatenate(axis=3)([x2, y3]))
    y1 = Decoder(64)(layers.Concatenate(axis=3)([x1, y2]))
    y0 = Decoder(64)(y1)

    # Hypercolumn
    y4 = layers.UpSampling2D(16, interpolation='bilinear')(y4)
    y3 = layers.UpSampling2D(8, interpolation='bilinear')(y3)
    y2 = layers.UpSampling2D(4, interpolation='bilinear')(y2)
    y1 = layers.UpSampling2D(2, interpolation='bilinear')(y1)
    hypercolumn = layers.Concatenate(axis=3)([y0, y1, y2, y3, y4])

    # Final conv
    outputs = keras.Sequential([
        layers.Conv2D(64, 3, padding='same', use_bias=False),
        layers.ELU(),
        layers.Conv2D(num_classes, 1, use_bias=False)
    ])(hypercolumn)

    outputs = tf.nn.softmax(outputs)
    return keras.Model(inputs, outputs)

if __name__ == '__main__':
    m = ResNet34UNetV2((160, 160, 3), 4)
    m.summary()
