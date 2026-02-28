import numpy as np
import keras

from srm_filter_kernel import all_normalized_hpf_list
from my_sequence import MySequence
from yedroudj_net import yedroudj_net
from yedroudj_net_64 import yedroudj_net_64
from ksrnet import ksrnet

def main():
    cover_labeled_files = []
    stego_labeled_files = []
    for i in range(1, DATASET_SIZE + 1):
        cover_labeled_files.append(f"{COVER_PATH}/{i}.{FORMAT}")
        stego_labeled_files.append(f"{STEGO_PATH}/{i}.{FORMAT}")
    test_files = np.asarray(cover_labeled_files[5_000:10_000] +
                            stego_labeled_files[5_000:10_000])
    test_labels = np.asarray([keras.utils.to_categorical(0, 2) for _ in range(5_000)] +
                            [keras.utils.to_categorical(1, 2) for _ in range(5_000)])
    test_indices = np.arange(10_000)
    test_files, test_labels = test_files[test_indices], test_labels[test_indices]
    test_sequence = MySequence(test_files, test_labels, batch_size=BATCH_SIZE, shuffle=False)

    model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    # model = ksrnet(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    model.compile(metrics=["accuracy"])

    min_test_loss = 314
    max_test_acc = -314

    for i in range(STEP, MAX_NUM + 1, STEP):
        num = str(i).zfill(4)
        model.load_weights(f"{DIRECTORY_NAME}/weights_epoch_{num}.h5")
        # model = keras.models.load_model(f"S-UNIWARD_0.4_yedroudj_net_64_cosine_decay/model_epoch_{num}")
        arr = model.evaluate(test_sequence)
        loss, acc = arr
        
        with open(OUTPUT_PATH, 'a') as f:
            f.write(num + ' ' + str(arr))
            f.write('\n')
        
        min_test_loss, max_test_acc = min(loss, min_test_loss), max(acc, max_test_acc)
        
    with open(OUTPUT_PATH, 'a') as f:
        f.write(str(min_test_loss) + ' ' + str(max_test_acc))
        f.write('\n')



COVER_PATH = "D:/Documents/DATASETS/BOSSbase_1.01_256x256"
STEGO_PATH = "D:/Documents/DATASETS/S-UNIWARD_256x256_0.4_bpp/stego_images"
FORMAT = "pgm"
OUTPUT_PATH = "S-UNI_256_0.4_yedroudj_net_64_cosine_decay_from_1e-2_to_1.5e-3_testing.txt"
DATASET_SIZE = 10_000
BATCH_SIZE = 32
DIRECTORY_NAME = "S-UNI_256_0.4_yedroudj_net_64_cosine_decay_from_1e-2_to_1.5e-3"
STEP = 5
MAX_NUM = 400

if __name__ == "__main__":
    main()
