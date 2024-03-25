from keras.src.legacy.preprocessing.image import ImageDataGenerator


class DataAugmentation:
    def __init__(self, config):
        """Initializes data augmentation with a given configuration."""
        self.config = config
        self.datagen = ImageDataGenerator(**config)

    def augment_data(self, X_train, y_train, batch_size):
        """Returns an augmented data generator."""
        return self.datagen.flow(X_train, y_train, batch_size=batch_size)
