import numpy as np
import tensorflow as tf

from keras import optimizers, losses
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger

from yedroudj_net import yedroudj_net
from yedroudj_net_64 import yedroudj_net_64
from ksrnet import ksrnet
from ksrnet64 import ksrnet64
from my_sequence import MySequence
from periodic_save_config import PeriodicSaveConfig
from periodic_full_save_config import PeriodicFullSaveConfig
from step_lr_schedule import StepLRSchedule
from srm_filter_kernel import all_normalized_hpf_list

import subprocess
import sys
import os

def main():
    COVER_PATHS = (
        "D:/Documents/DATASETS/BOSSbase_1.01_256x256/cover_images",
        "D:/Documents/DATASETS/BOWS2/cover_images"
    )
    STEGO_PATHS = (
        "D:/Documents/DATASETS/BOSSbase_1.01_256x256/WOW_256x256_0.2_bpp/stego_images",
        "D:/Documents/DATASETS/BOWS2/WOW_256x256_0.2_bpp/stego_images"
    )
    IMAGE_FORMAT = "pgm"
    DATASET_SIZE = 10_000
    EPOCHS = 400
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.06667
    
    cover_labeled_files = []
    stego_labeled_files = []
    
    for cover_path, stego_path in zip(COVER_PATHS, STEGO_PATHS):
        for i in range(1, DATASET_SIZE + 1):
            cover_labeled_files.append(f"{cover_path}/{i}.{IMAGE_FORMAT}")
            stego_labeled_files.append(f"{stego_path}/{i}.{IMAGE_FORMAT}")
    
    files = np.asarray(cover_labeled_files[:5_000] + cover_labeled_files[10_000:20_000] +
                       stego_labeled_files[:5_000] + stego_labeled_files[10_000:20_000])
    labels = np.asarray([to_categorical(0, 2) for _ in range(15_000)] +
                        [to_categorical(1, 2) for _ in range(15_000)])
    
    files_size = len(files)
    indices = np.arange(files_size)
    
    np.random.seed(314)
    np.random.shuffle(indices)
    
    validation_size = int(files_size * VALIDATION_SPLIT)
    train_indices = indices[validation_size:]
    validation_indices = indices[:validation_size]
    
    train_files, train_labels = files[train_indices], labels[train_indices]
    validation_files, validation_labels = files[validation_indices], labels[validation_indices]

    train_sequence = MySequence(train_files, train_labels, batch_size=BATCH_SIZE, shuffle=False)
    validation_sequence = MySequence(validation_files, validation_labels, batch_size=BATCH_SIZE, shuffle=False)
    
    model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    model_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.01,
        decay_steps=100_000,
        alpha=0.1
    )
    model_optimizer = optimizers.SGD(
        learning_rate=model_schedule,
        momentum=0.95
    )
    
    model_loss = losses.CategoricalCrossentropy()
    model.compile(
        optimizer=model_optimizer,
        loss=model_loss,
        metrics=["accuracy"]
    )
    
    dirpath = "BOWS2_WOW_256_0.2_yedroudj_net_64_cosine_decay_1"
    os.makedirs(dirpath, exist_ok=True)
    
    weights_saver = PeriodicSaveConfig(dirpath=dirpath, period=5)
    
    print(model.summary())
    
    model.fit(train_sequence,
              validation_data=validation_sequence,
              epochs=EPOCHS,
              shuffle=False,
              callbacks=[weights_saver],
    )
    
    subprocess.run([sys.executable, 'find_best_test_accuracy.py'])



if __name__ == "__main__":
    main()
