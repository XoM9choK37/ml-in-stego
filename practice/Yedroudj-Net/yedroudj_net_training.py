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
# import json
# import time
# # import random

# # os.environ['TF_DETERMINISTIC_OPS'] = '1'
# # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# # seed = 314159

# # tf.keras.utils.set_random_seed(seed)
# # tf.config.experimental.enable_op_determinism()

# # np.random.seed(seed)
# # random.seed(seed)

def main():
    COVER_PATH = "D:/Documents/DATASETS/BOSSbase_1.01_256x256/cover_images"
    # STEGO_PATH = "D:/Documents/DATASETS/BOSSbase_1.01_256x256/S-UNIWARD_256x256_0.2_bpp/stego_images"
    STEGO_PATH = "D:/Documents/DATASETS/BOSSbase_1.01_256x256/WOW_256x256_0.5_bpp/stego_images"
    IMAGE_FORMAT = "pgm"
    DATASET_SIZE = 10_000
    EPOCHS = 400
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.20
    
    cover_labeled_files = []
    stego_labeled_files = []
    
    for i in range(1, DATASET_SIZE + 1):
        cover_labeled_files.append(f"{COVER_PATH}/{i}.{IMAGE_FORMAT}")
        stego_labeled_files.append(f"{STEGO_PATH}/{i}.{IMAGE_FORMAT}")
    
    files = np.asarray(cover_labeled_files[:5_000] +
                       stego_labeled_files[:5_000])
    labels = np.asarray([to_categorical(0, 2) for _ in range(5_000)] +
                        [to_categorical(1, 2) for _ in range(5_000)])
    # test_files = np.asarray(cover_labeled_files[5_000:10_000] +
    #                         stego_labeled_files[5_000:10_000])
    # test_labels = np.asarray([to_categorical(0, 2) for _ in range(5_000)] +
    #                          [to_categorical(1, 2) for _ in range(5_000)])
    
    files_size = len(files)
    indices = np.arange(files_size)
    # test_files_size = len(test_files)
    # test_indices = np.arange(test_files_size)
    
    np.random.seed(314)
    np.random.shuffle(indices)
    
    validation_size = int(files_size * VALIDATION_SPLIT)
    train_indices = indices[validation_size:]
    validation_indices = indices[:validation_size]
    
    train_files, train_labels = files[train_indices], labels[train_indices]
    validation_files, validation_labels = files[validation_indices], labels[validation_indices]
    # test_files, test_labels = test_files[test_indices], test_labels[test_indices]

    train_sequence = MySequence(train_files, train_labels, batch_size=BATCH_SIZE, shuffle=False)
    validation_sequence = MySequence(validation_files, validation_labels, batch_size=BATCH_SIZE, shuffle=False)
    # test_sequence = MySequence(test_files, test_labels, batch_size=BATCH_SIZE, shuffle=False)

    # model = yedroudj_net(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    
    # model_optimizer = optimizers.SGD(
    #     learning_rate=StepLRSchedule(0.01, 40500, 0.1, np.ceil(files_size / BATCH_SIZE)),
    #     momentum=0.95
    # )
    
    # model_optimizer = optimizers.adamw_experimental.AdamW(learning_rate=1e-3)
    
    # model_optimizer = optimizers.adamw_experimental.AdamW(learning_rate=1e-5)
    
    # model_optimizer = optimizers.adamw_experimental.AdamW(learning_rate=1e-7)
    
    # model_optimizer = optimizers.Adamax(learning_rate=1e-5)
    
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=100_000,
    #     alpha=0.0
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95
    # )
    
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=1e-4,
    #     decay_steps=100_000,
    #     alpha=0.01
    # )
    # model_optimizer = optimizers.adamw_experimental.AdamW(
    #     learning_rate=model_scheduler
    # )
    
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=100_000,
    #     alpha=0.01
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95
    # )
    
    # model_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    #     initial_learning_rate=0.01,
    #     first_decay_steps=25_000,
    #     t_mul=2.0,
    #     m_mul=0.5,
    #     alpha=0.01
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_schedule,
    #     momentum=0.95
    # )
    
    # model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_optimizer = optimizers.SGD(
    #     learning_rate=StepLRSchedule(0.01, 40500, 0.1, np.ceil(files_size / BATCH_SIZE)),
    #     momentum=0.95
    # )
    
    # model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=100_000,
    #     alpha=0.1
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95
    # )
    
    # model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=100_000,
    #     alpha=0.01
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95
    # )
    
    # model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
    #     initial_learning_rate=0.01,
    #     first_decay_steps=25_000,
    #     t_mul=2.0,
    #     m_mul=0.5,
    #     alpha=0.1
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_schedule,
    #     momentum=0.95
    # )
    
    # model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.1,
    #     decay_steps=85_000,
    #     alpha=0.015
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95
    # )
    
    # model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=85_000,
    #     alpha=0.15
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95
    # )
    
    # model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=100_000,
    #     alpha=0.15
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95
    # )
    
    # # model = ksrnet64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    # #     initial_learning_rate=0.01,
    # #     decay_steps=100_000,
    # #     alpha=0.1
    # # )
    # # model_optimizer = optimizers.SGD(
    # #     learning_rate=model_scheduler,
    # #     momentum=0.95
    # # )
    
    # # model = yedroudj_net(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # # model_optimizer = optimizers.SGD(
    # #     learning_rate=StepLRSchedule(0.01, 40500, 0.1, np.ceil(files_size / BATCH_SIZE)),
    # #     momentum=0.95
    # # )
    
    # model_loss = losses.CategoricalCrossentropy()
    
    # model.compile(
    #     optimizer=model_optimizer,
    #     loss=model_loss,
    #     metrics=["accuracy"]
    # )
    
    # print(model.summary())
    
    # model_saver = PeriodicFullSaveConfig(
    #     dirpath='S-UNIWARD_0.4_yedroudj_net_64_cosine_decay',
    #     period=10,
    #     model_name_template='model_epoch_{epoch:04d}',
    #     save_format='tf',
    #     save_weights_also=True,
    #     verbose=1
    # )
    
    # # start = time.time()
    # # model.fit(train_sequence,
    # #           validation_data=validation_sequence,
    # #           epochs=EPOCHS,
    # #           shuffle=False,
    # #           callbacks=[PeriodicSaveConfig(dirpath="S-UNI_256_0.4_yedroudj_net_classic", period=5)]
    # # )
    # # end = time.time()
    # # print(start)
    # # print(end)
    # # print(end - start)
    # # with open("yedroudj_net_classic_time.txt", 'a') as f:
    # #     f.write(str(start))
    # #     f.write('\n')
    # #     f.write(str(end))
    # #     f.write('\n')
    # #     f.write(str(end - start))
    
    # model.fit(train_sequence,
    #           validation_data=validation_sequence,
    #           epochs=EPOCHS,
    #           shuffle=False,
    #           callbacks=[model_saver]
    # )
    
    # model = ksrnet64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=100_000,
    #     alpha=0.1
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95
    # )
    
    # # model = ksrnet(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # # model_optimizer = optimizers.SGD(
    # #     learning_rate=StepLRSchedule(0.01, 40500, 0.1, np.ceil(files_size / BATCH_SIZE)),
    # #     momentum=0.95
    # # )
    
    # model = ksrnet64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=100_000,
    #     alpha=0.1
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95
    # )
    
    # model = ksrnet(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=100_000,
    #     alpha=0.1
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95
    # )
    
    # model = ksrnet64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=100_000,
    #     alpha=0.1
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_scheduler,
    #     momentum=0.95,
    #     nesterov=True
    # )
    
    # model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model_schedule = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=0.01,
    #     decay_steps=100_000,
    #     alpha=0.1
    # )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=model_schedule,
    #     momentum=0.95
    # )
    
    # model_loss = losses.CategoricalCrossentropy()
    
    # model.compile(
    #     optimizer=model_optimizer,
    #     loss=model_loss,
    #     metrics=["accuracy"]
    # )
    
    # # # model = yedroudj_net(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # # # model_optimizer = optimizers.SGD(
    # # #     learning_rate=StepLRSchedule(0.01, 40500, 0.1, np.ceil(files_size / BATCH_SIZE)),
    # # #     momentum=0.95
    # # # )
    
    model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    model_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.001,
        decay_steps=100_000,
        alpha=0.001
    )
    model_optimizer = optimizers.SGD(
        learning_rate=model_schedule,
        momentum=0.95
    )
    # model_optimizer = optimizers.SGD(
    #     learning_rate=StepLRSchedule(0.01, 22500, 0.1, 250),
    #     momentum=0.95
    # )
    
    model_loss = losses.CategoricalCrossentropy()
    model.compile(
        optimizer=model_optimizer,
        loss=model_loss,
        metrics=["accuracy"]
    )
    
    dirpath = "WOW_256_0.5_yedroudj_net_64_cosine_decay_4_not_norm"
    os.makedirs(dirpath, exist_ok=True)
    
    # optimizer_info = {
    #     "optimizer_name": model.optimizer.__class__.__name__,
    #     "optimizer_config": model.optimizer.get_config(),
    #     "learning_rate_value": tf.keras.backend.get_value(model.optimizer.learning_rate),
    #     "learning_rate_schedule": str(type(model_optimizer.learning_rate).__name__) if hasattr(model_optimizer, 'learning_rate') else None,
    #     "schedule_config": model_optimizer.learning_rate.get_config() if hasattr(model_optimizer.learning_rate, 'get_config') else None,
    #     "steps_per_epoch": len(train_sequence),
    #     "total_epochs": 400
    # }
    # with open(os.path.join(dirpath, "optimizer_info.json"), "w", encoding="utf-8") as f:
    #     json.dump(optimizer_info, f, indent=2, ensure_ascii=False, default=str)
    
    weights_saver = PeriodicSaveConfig(dirpath=dirpath, period=5)
    
    # weights_saver = ModelCheckpoint(
    #     filepath=os.path.join(dirpath, "weights_epoch_{epoch:04d}.h5"),
    #     save_weights_only=True,
    #     save_freq=5 * 250,
    #     verbose=1
    # )
    # csv_logger = CSVLogger(
    #     filename=os.path.join(dirpath, "training_log.csv"),
    #     separator=',',
    #     append=False
    # )
    
    print(model.summary())
    
    # # start = time.time()
    model.fit(train_sequence,
              validation_data=validation_sequence,
              epochs=EPOCHS,
              shuffle=False,
              callbacks=[weights_saver],
    )
    # # end = time.time()
    # # print(start)
    # # print(end)
    # # print(end - start)
    # # with open(os.path.join(dirpath, "__training_time__.txt"), 'a') as f:
    # #     f.write(str(start))
    # #     f.write('\n')
    # #     f.write(str(end))
    # #     f.write('\n')
    # #     f.write(str(end - start))
    
    # # model.evaluate(test_sequence)
    
    subprocess.run([sys.executable, 'find_best_test_accuracy.py'])



if __name__ == "__main__":
    main()
