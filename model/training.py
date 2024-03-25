from keras.src.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.src.utils import to_categorical
from sklearn.model_selection import train_test_split

from model.custom_densenet169 import CustomDenseNet
from utils.data_augmentation import DataAugmentation
from utils.dataset import CustomDataset
from utils.loader import load_json_config
from utils.preprocessor import DataPreprocessor

import argparse

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

default_conf_path = 'conf/conf.json'


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train CustomDenseNet on COVID-19 dataset.")
    parser.add_argument('--config', type=str, default=default_conf_path, help='Path to configuration file.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    config = load_json_config(args.config)

    data_dir = config['data_dir']
    categories = config['categories']
    model_save_path = config['model_save_path']
    target_size = config['target_size']
    batch_size = config['batch_size']
    epochs = config['epochs']
    num_classes = len(categories)
    rotation_range = config['rotation_range'],
    width_shift_range = config['width_shift_range'],
    height_shift_range = config['height_shift_range'],
    shear_range = config['shear_range'],
    zoom_range = config['zoom_range'],
    horizontal_flip = config['horizontal_flip'],
    fill_mode = config['fill_mode']

    logging.info(f"data_dir: {data_dir}")
    logging.info(f"categories: {categories}")
    logging.info(f"model_save_path: {model_save_path}")
    logging.info(f"target_size: {target_size}")
    logging.info(f"batch_size: {batch_size}")
    logging.info(f"epochs: {epochs}")
    logging.info(f"num_classes: {num_classes}")
    logging.info(f"rotation_range: {rotation_range}"),
    logging.info(f"width_shift_range: {width_shift_range}"),
    logging.info(f"height_shift_range: {height_shift_range}"),
    logging.info(f"shear_range: {shear_range}"),
    logging.info(f"zoom_range: {zoom_range}"),
    logging.info(f"horizontal_flip: {horizontal_flip}"),
    logging.info(f"fill_mode: {fill_mode}")

    preprocessor = DataPreprocessor(data_dir, target_size=target_size)
    data, labels = preprocessor.load_images(categories)

    X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    train_dataset = CustomDataset(X_train, y_train, batch_size)
    val_dataset = CustomDataset(X_val, y_val, batch_size)

    logging.log(logging.INFO, f"shape of y_train: {y_train.shape}")
    logging.log(logging.INFO, f"shape of y_val: {y_val.shape}")

    augmentation_config = {
        "rotation_range": config['rotation_range'],
        "width_shift_range": config['width_shift_range'],
        "height_shift_range": config['height_shift_range'],
        "shear_range": config['shear_range'],
        "zoom_range": config['zoom_range'],
        "horizontal_flip": config['horizontal_flip'],
        "fill_mode": config['fill_mode']
    }

    data_augmentation = DataAugmentation(augmentation_config)

    data_augmentation.datagen.fit(X_train)

    model = CustomDenseNet(num_classes=num_classes, input_shape=target_size + [3])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    model.fit(train_dataset,
              validation_data=val_dataset,
              epochs=epochs,
              callbacks=[checkpoint, reduce_lr, early_stopping])

    logging.info("Training completed.")
