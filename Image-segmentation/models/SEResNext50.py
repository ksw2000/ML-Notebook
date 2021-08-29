import tensorflow.keras as keras
import tensorflow as tf
from keras import layers, models

class SEBlock(keras.Model):
    def __init__(self, ratio=16):
        super().__init__()
        self.ratio = ratio
        self.gap = layers.GlobalAveragePooling2D()

    def build(self, input_shape):
        filters = input_shape[-1]
        self.reshape = layers.Reshape((1, 1, filters))
        self.fc1 = layers.Dense(
            filters // self.ratio, kernel_initializer='he_normal', use_bias=False, activation='relu')
        self.fc2 = layers.Dense(
            filters, kernel_initializer='he_normal', use_bias=False, activation='sigmoid')

    def call(self, input):
        x = self.gap(input)
        x = self.reshape(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return tf.multiply(x, input)
    
    def get_config(self):
        return{
            'ratio': self.ratio
        }
    

class SEResNeXtUnit(keras.Model):
    def __init__(self, filters, strides, cardinality=32):
        super().__init__()
        self.filters = filters
        self.strides = strides

        self.conv1x1_1 = layers.Conv2D(filters, 1, 1)
        self.bn1 = layers.BatchNormalization()

        self.conv3x3 = layers.Conv2D(
            filters, 3, strides, groups=cardinality, padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv1x1_2 = layers.Conv2D(filters*2, 1, 1)
        self.bn3 = layers.BatchNormalization()

        self.seBlock = SEBlock()

    def build(self, input_shape):
        input_filters = input_shape[-1]
        if input_filters != self.filters*2:
            self.shortcut = models.Sequential([
                layers.Conv2D(self.filters*2, 1, self.strides),
                layers.BatchNormalization()
            ])
        else:
            self.shortcut = models.Sequential()

    def call(self, input):
        x = input
        x = self.conv1x1_1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv3x3(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)

        x = self.conv1x1_2(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)

        # Add SEBlock here
        x = self.seBlock(x)

        shortcut = self.shortcut(input)
        return tf.nn.relu(tf.add(x, shortcut))
    
    def get_config(self):
        return{
            'filters': self.filters,
            'strides': self.strides
        }

class Encoder(keras.Model):
    def __init__(self, channels, repeat, strides):
        super().__init__()
        self.resBlocks = keras.Sequential()
        self.resBlocks.add(SEResNeXtUnit(channels, strides))
        for _ in range(1, repeat):
            self.resBlocks.add(SEResNeXtUnit(channels, strides=1))

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
        self.cbam = CBAM(reduction=16)

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
        x = self.cbam(x)

        shortcut = self.conv1x1(self.upsample(inputs))
        x += shortcut
        return x

    def get_config(self):
        return {}


def SEResNeXt50UNet(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Encode by ResNet34
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x0 = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # ResNet34
    x1 = Encoder(128, 3, strides=1)(x0)
    x2 = Encoder(256, 4, strides=2)(x1)
    x3 = Encoder(512, 6, strides=2)(x2)
    x4 = Encoder(1024, 3, strides=2)(x3)

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
    m = SEResNeXt50UNet((160, 160, 3), 4)
    m.summary()
