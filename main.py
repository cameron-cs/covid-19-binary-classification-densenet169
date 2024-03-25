import argparse

import logging

import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report

from utils.loader import load_json_config
from utils.preprocessor import DataPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

main_conf_path = 'conf/conf_main.json'


def parse_arguments():
    parser = argparse.ArgumentParser(description='Run inference on a saved model with given configuration.')
    parser.add_argument('--config', type=str, default=main_conf_path, help='Path to configuration JSON file.')

    return parser.parse_args()


if __name__ == '__main__':
    config = load_json_config(parse_arguments().config)

    model_path = config['model_path']
    data_dir = config['data_dir']
    target_size = tuple(config['target_size'])
    batch_size = config['batch_size']

    logging.info(f"model_path: {model_path}")
    logging.info(f"data_dir: {data_dir}")
    logging.info(f"target_size: {target_size}")
    logging.info(f"batch_size: {batch_size}")

    try:
        # Load the saved model
        model = load_model(model_path)

        # Preprocess the data
        preprocessor = DataPreprocessor(data_dir, target_size)
        test_data, test_labels = preprocessor.load_images(categories=['COVID', 'non-COVID'])

        # Making predictions
        predictions = model.predict(test_data)
        predicted_labels = np.argmax(predictions, axis=1)

        # Ensure test_labels are integers if they're already not in one-hot encoding
        if test_labels.ndim == 2:
            test_labels = np.argmax(test_labels, axis=1)

        # Evaluating the model
        report = classification_report(test_labels, predicted_labels, target_names=['COVID', 'non-COVID'])
        logging.info(f"Classification report: \n{report}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

