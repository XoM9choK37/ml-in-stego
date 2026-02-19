import numpy as np

from keras import optimizers, losses
from keras.utils import to_categorical

from yedroudj_net import yedroudj_net
from my_sequence import MySequence
from periodic_save_config import PeriodicSaveConfig
from step_lr_schedule import StepLRSchedule
from srm_filter_kernel import all_normalized_hpf_list

def main():
    COVER_PATH = "F:/MSI Katana/D/DATASETS/BOSSbase_1.01_256x256"
    STEGO_PATH = "F:/MSI Katana/D/DATASETS/S-UNIWARD_256x256_0.4_bpp/stego_images"
    FORMAT = "pgm"
    DATASET_SIZE = 10_000
    EPOCHS = 400
    BATCH_SIZE = 32
    VALIDATION_SPLIT = 0.20
    
    cover_labeled_files = []
    stego_labeled_files = []
    for i in range(1, DATASET_SIZE + 1):
        cover_labeled_files.append(f"{COVER_PATH}/{i}.{FORMAT}")
        stego_labeled_files.append(f"{STEGO_PATH}/{i}.{FORMAT}")
    
    files = np.asarray(cover_labeled_files[:5_000] +
                       stego_labeled_files[:5_000])
    labels = np.asarray([to_categorical(0, 2) for _ in range(5_000)] +
                        [to_categorical(1, 2) for _ in range(5_000)])
    test_files = np.asarray(cover_labeled_files[5_000:10_000] +
                            stego_labeled_files[5_000:10_000])
    test_labels = np.asarray([to_categorical(0, 2) for _ in range(5_000)] +
                             [to_categorical(1, 2) for _ in range(5_000)])
    
    files_size = len(files)
    indices = np.arange(files_size)
    test_files_size = len(test_files)
    test_indices = np.arange(test_files_size)
    
    np.random.seed(314)
    np.random.shuffle(indices)
    
    validation_size = int(files_size * VALIDATION_SPLIT)
    train_indices = indices[validation_size:]
    validation_indices = indices[:validation_size]
    
    train_files, train_labels = files[train_indices], labels[train_indices]
    validation_files, validation_labels = files[validation_indices], labels[validation_indices]
    test_files, test_labels = test_files[test_indices], test_labels[test_indices]

    train_sequence = MySequence(train_files, train_labels, batch_size=BATCH_SIZE, shuffle=False)
    validation_sequence = MySequence(validation_files, validation_labels, batch_size=BATCH_SIZE, shuffle=False)
    test_sequence = MySequence(test_files, test_labels, batch_size=BATCH_SIZE, shuffle=False)

    model = yedroudj_net(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    print(model.summary())
    
    # model_optimizer = optimizers.SGD(
    #     learning_rate=StepLRSchedule(0.01, 40500, 0.1, np.ceil(files_size / BATCH_SIZE)),
    #     momentum=0.95
    # )
    model_optimizer = optimizers.adamw_experimental.AdamW()
    model_loss = losses.CategoricalCrossentropy()
    
    model.compile(
        optimizer=model_optimizer,
        loss=model_loss,
        metrics=["accuracy"]
    )
    
    model.fit(train_sequence,
              validation_data=validation_sequence,
              epochs=EPOCHS,
              shuffle=False,
              callbacks=[PeriodicSaveConfig(dirpath="yedroudj_net_adamw", period=5)]
    )
    
    model.evaluate(test_sequence)



if __name__ == "__main__":
    main()
