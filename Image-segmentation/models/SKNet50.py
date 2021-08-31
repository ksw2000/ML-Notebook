import tensorflow.keras as keras
import tensorflow as tf
from keras import layers, models


class SKUnit(keras.Model):
    def __init__(self, filters, strides, M=2, G=32, r=16, L=32):
        """ Constructor
        Args:
            M: the number of branchs.
            G: num of convolution groups.
            r: the ratio for compute d, the length of z.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super().__init__()
        self.M = M
        self.filters = filters

        self.convs = []  # 各分支的卷積層
        for i in range(M):
            self.convs.append(models.Sequential([
                layers.Conv2D(filters, 3+2*i, strides,
                              padding='same', groups=G),
                layers.BatchNormalization(),
                layers.Activation('relu'),
            ]))

        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(max(filters//r, L))
        self.fcs = []
        for i in range(M):
          self.fcs.append(layers.Dense(filters))

    def call(self, input):
        # 計算不同分支的 U
        for i in range(self.M):
            feat = self.convs[i](input)
            feat = tf.expand_dims(feat, axis=-1)
            feats_U = feat if i == 0 else tf.concat([feats_U, feat], axis=-1)

        # feats_U (H, W, filters, M)

        # 對 U 做全局平均池化得到 s
        feats_s = self.gap(tf.reduce_sum(feats_U, axis=-1))

        # s 經過全連結層可以得到 z
        feats_Z = self.fc(feats_s)

        for i in range(self.M):
            fcs = self.fcs[i](feats_Z)
            att_vec = fcs if i == 0 else tf.concat([att_vec, fcs], axis=-1)

        att_vec = layers.Reshape((1, 1, self.filters, self.M))(att_vec)
        att_vec = tf.nn.softmax(att_vec, axis=-1)

        # att_vec (1, 1, filters, M)

        mul = tf.multiply(feats_U, att_vec)

        return tf.reduce_sum(mul, axis=-1)


class SKConv(keras.Model):
    def __init__(self, filters, strides, M=2, G=32, r=16, L=32):
        super().__init__()
        self.filters = filters
        self.strides = strides

        #----------------------------- conv1x1_1 -----------------------------
        self.conv1x1_1 = layers.Conv2D(filters, 1, 1)
        self.bn1 = layers.BatchNormalization()

        #------------------------------ middle -------------------------------
        self.skunit = SKUnit(filters, strides, M, G, r, L)
        self.bn2 = layers.BatchNormalization()

        #----------------------------- conv1x1_2 -----------------------------
        self.conv1x1_2 = layers.Conv2D(filters*2, 1, 1)
        self.bn3 = layers.BatchNormalization()

    def build(self, input_shape):
        if input_shape[-1] != self.filters*2:
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

        x = self.skunit(x)
        x = self.bn2(x)

        x = self.conv1x1_2(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)

        shortcut = self.shortcut(input)

        return tf.nn.relu(tf.add(x, shortcut))

class Encoder(keras.Model):
    def __init__(self, filters, strides, repeat, M=2, G=32, r=16, L=32):
        super().__init__()
        self.resBlocks = keras.Sequential()
        self.resBlocks.add(SKConv(filters, strides, M, G, r, L))
        for _ in range(1, repeat):
            self.resBlocks.add(
                SKConv(filters, 1, M, G, r, L))

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


def SKNet50UNet(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Encode by SKNet50
    x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = tf.nn.relu(x)
    x0 = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # SKNet50
    x1 = Encoder(filters=128, strides=1, repeat=3)(x0)
    x2 = Encoder(filters=256, strides=2, repeat=4)(x1)
    x3 = Encoder(filters=512, strides=2, repeat=6)(x2)
    x4 = Encoder(filters=1024, strides=2, repeat=5)(x3)

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
    m = SKNet50UNet((160, 160, 3), 4)
    m.summary()
