import numpy as np
from keras.utils import load_img, Sequence

class MySequence(Sequence):
    def __init__(self, files, labels, batch_size=32, shuffle=True):
        self.files = np.asarray(files)
        self.labels = np.asarray(labels)
        self.batch_size = batch_size
        self.indices = np.arange(len(self.files))
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.indices)
    def __len__(self):
        return int(np.ceil(len(self.files) / self.batch_size))
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_files = self.files[batch_indices]
        batch_labels = self.labels[batch_indices]
        batch_images = np.zeros((len(batch_files), 256, 256, 1), dtype=np.float32)
        for index, file in enumerate(batch_files):
            image = load_img(file, color_mode="grayscale")
            array = np.asarray(image, dtype=np.float32) / 255.0
            batch_images[index, ..., 0] = array
        return batch_images, np.asarray(batch_labels, dtype=np.float32)
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
