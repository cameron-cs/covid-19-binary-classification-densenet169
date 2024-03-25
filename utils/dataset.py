from keras.utils import Sequence


class CustomDataset(Sequence):
    def __init__(self, data, labels, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.data = data
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data) // self.batch_size

    def __getitem__(self, idx):
        # logic to get a single batch at index `idx`
        batch_x = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y
