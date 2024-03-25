import cv2
import numpy as np
import os
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import LabelEncoder


class DataPreprocessor:
    def __init__(self, data_dir, target_size=(224, 224)):
        self.data_dir = data_dir
        self.target_size = target_size
        self.label_encoder = LabelEncoder()  # To encode label strings to integers

    def preprocess_image(self, filepath):
        """Loads and preprocesses an image."""
        image = cv2.imread(filepath)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            image = image / 255.0  # Normalize to [0,1]
            return image
        else:
            return None

    def load_images(self, categories, max_workers=4):
        """Loads and preprocesses images from given directory categories with parallel processing."""
        data = []
        labels = []

        def process_file(category, file):
            filepath = os.path.join(self.data_dir, category, file)
            if filepath.lower().endswith(('.png', '.jpg', '.jpeg')) and not file.startswith('.'):
                image = self.preprocess_image(filepath)
                if image is not None:
                    data.append(image)
                    labels.append(category)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for category in categories:
                category_dir = os.path.join(self.data_dir, category)
                for file in os.listdir(category_dir):
                    if not file.startswith('.'):  # Skip system files like .DS_Store
                        executor.submit(process_file, category, file)

        # Fit and transform labels to integers
        int_labels = self.label_encoder.fit_transform(labels)

        return np.array(data), int_labels

    def split_data(self, data, labels, test_size=0.2, random_state=7):
        """Splits data into training and test (optionally validation) sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=random_state)
        # Apply one-hot encoding to the integer labels
        y_train = to_categorical(y_train, num_classes=len(self.label_encoder.classes_))
        y_test = to_categorical(y_test, num_classes=len(self.label_encoder.classes_))
        return X_train, X_test, y_train, y_test