from keras import Model, layers
from keras.applications import DenseNet169


class CustomDenseNet(Model):
    def __init__(self, num_classes=2, input_shape=(224, 224, 3), **kwargs):
        super(CustomDenseNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.base_model = DenseNet169(include_top=False, weights='imagenet', input_shape=input_shape)
        self.base_model.trainable = False  # freeze the base model
        self.global_pool = layers.GlobalAveragePooling2D()
        self.batch_norm1 = layers.BatchNormalization()
        self.dense1 = layers.Dense(64, activation='relu')
        self.batch_norm2 = layers.BatchNormalization()
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.base_model(inputs, training=training)
        x = self.global_pool(x)
        x = self.batch_norm1(x, training=training)
        x = self.dense1(x)
        x = self.batch_norm2(x, training=training)
        return self.output_layer(x)

    def get_config(self):
        config = super(CustomDenseNet, self).get_config()
        config.update({
            'num_classes': self.num_classes,
            'input_shape': self.input_shape
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)