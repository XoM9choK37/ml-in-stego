import numpy as np
import keras

from yedroudj_net import yedroudj_net
from yedroudj_net_64 import yedroudj_net_64
from my_sequence import MySequence
from srm_filter_kernel import all_normalized_hpf_list

def main(cover_path, stego_path, image_format):
    cover_labeled_files = []
    stego_labeled_files = []
    for i in range(1, DATASET_SIZE + 1):
        cover_labeled_files.append(f"{cover_path}/{i}.{image_format}")
        stego_labeled_files.append(f"{stego_path}/{i}.{image_format}")
    test_files = np.asarray(cover_labeled_files[5_000:10_000] +
                            stego_labeled_files[5_000:10_000])
    test_labels = np.asarray([keras.utils.to_categorical(0, 2) for _ in range(5_000)] +
                            [keras.utils.to_categorical(1, 2) for _ in range(5_000)])
    test_files_size = len(test_files)
    test_indices = np.arange(test_files_size)
    test_files, test_labels = test_files[test_indices], test_labels[test_indices]
    test_sequence = MySequence(test_files, test_labels, batch_size=BATCH_SIZE, shuffle=False)

    model = yedroudj_net_64(input_shape=(256, 256, 1), all_normalized_hpf_list=all_normalized_hpf_list)
    model.compile(metrics=["accuracy"])

    model.load_weights(f"{DIRECTORY_NAME}/weights_epoch_{WEIGHTS_NUMBER}.h5")

    print(cover_path, stego_path, image_format)
    model.evaluate(test_sequence)



COVER_PATH = "D:/Documents/DATASETS/BOSSbase_1.01_256x256"
STEGO_PATH = "D:/Documents/DATASETS/S-UNIWARD_256x256_0.4_bpp/stego_images"
IMAGE_FORMAT = "pgm"
DATASET_SIZE = 10_000
BATCH_SIZE = 1
DIRECTORY_NAME = "S-UNI_256_0.4_yedroudj_net_64_cosine_decay_from_1e-2_to_1e-3"
WEIGHTS_NUMBER = "0290"

if __name__ == "__main__":
    paths = [
        ("../BOSSbase_1.01_256x256", "../S-UNIWARD_256x256_0.1_bpp/stego_images", "pgm"),
        ("../BOSSbase_1.01_256x256", "../S-UNIWARD_256x256_0.2_bpp/stego_images", "pgm"),
        ("../BOSSbase_1.01_256x256", "../S-UNIWARD_256x256_0.3_bpp/stego_images", "pgm"),
        ("../BOSSbase_1.01_256x256", "../S-UNIWARD_256x256_0.4_bpp/stego_images", "pgm"),
        ("../BOSSbase_1.01_256x256", "../S-UNIWARD_256x256_0.5_bpp/stego_images", "pgm"),
        ("../BOSSbase_1.01_bmp_256x256", "../SHELL_256x256_steghide_Anubis", "bmp"),
        ("../BOSSbase_1.01_bmp_256x256", "../SHELL_256x256_steghide_AWFULSHRED", "bmp"),
        ("../BOSSbase_1.01_bmp_256x256", "../SHELL_256x256_steghide_DarkRadiation", "bmp"),
        ("../BOSSbase_1.01_bmp_256x256", "../SHELL_256x256_steghide_IRCbot", "bmp"),
        ("../BOSSbase_1.01_bmp_256x256", "../SHELL_256x256_stegano_Anubis", "bmp"),
        ("../BOSSbase_1.01_bmp_256x256", "../SHELL_256x256_stegano_AWFULSHRED", "bmp"),
        ("../BOSSbase_1.01_bmp_256x256", "../SHELL_256x256_stegano_DarkRadiation", "bmp"),
        ("../BOSSbase_1.01_bmp_256x256", "../SHELL_256x256_stegano_IRCbot", "bmp"),
        ("../BOSSbase_1.01_bmp_256x256", "../SHELL_256x256", "bmp"),
        (COVER_PATH, STEGO_PATH, IMAGE_FORMAT)
    ]
    for cover_path, stego_path, image_format in paths[-1:]:
        main(cover_path, stego_path, image_format)

## Training & testing \/ \/ \/

## ckpt1/0400.h5 — min_loss: 0.2999 (among ckpt1)
## ckpt1/0400.h5 — max_accuracy: 0.8702 (among ckpt1)
## ckpt1/0115.h5 — min_val_loss: 0.6206 (among ckpt1)
## ckpt1/0190.h5 — max_val_accuracy: 0.6705 (among ckpt1)
## ckpt1/0085.h5 — min_test_loss: 0.0262 (among ckpt1, testing on S-UNI_256_0.4)
## ckpt1/0380.h5 — max_test_accuracy: 0.7754 (among ckpt1, testing on S-UNI_256_0.4)

## Testing \/ \/ \/

## S-UNIWARD_256x256_0.1_bpp -> ckpt1/0380.h5 — accuracy: 0.5127    ksr3_n_0020(t:335)_and_below/// 0.5137
## S-UNIWARD_256x256_0.2_bpp -> ckpt1/0380.h5 — accuracy: 0.5748    ksr/// 0.5826
## S-UNIWARD_256x256_0.3_bpp -> ckpt1/0380.h5 — accuracy: 0.6896    ksr/// 0.6877
## S-UNIWARD_256x256_0.4_bpp -> ckpt1/0380.h5 — accuracy: 0.7754    ksr/// 0.7679
## S-UNIWARD_256x256_0.5_bpp -> ckpt1/0380.h5 — accuracy: 0.8199    ksr/// 0.8108

## SHELL_256x256_steghide_Anubis -> ckpt1/0380.h5 — accuracy: 0.6481    ksr/// 0.6522
## SHELL_256x256_steghide_AWFULSHRED -> ckpt1/0380.h5 — accuracy: 0.7143    ksr/// 0.7070
## SHELL_256x256_steghide_DarkRadiation -> ckpt1/0380.h5 — accuracy: 0.7980    ksr/// 0.7830
## SHELL_256x256_steghide_IRCbot -> ckpt1/0380.h5 — accuracy: 0.6892    ksr/// 0.6854

## SHELL_256x256_stegano_Anubis -> ckpt1/0380.h5 — accuracy: 0.5542    ksr/// 0.5455
## SHELL_256x256_stegano_AWFULSHRED -> ckpt1/0380.h5 — accuracy: 0.7652    ksr/// 0.7447
## SHELL_256x256_stegano_DarkRadiation -> ckpt1/0380.h5 — accuracy: 0.7828    ksr/// 0.7791
## SHELL_256x256_stegano_IRCbot -> ckpt1/0380.h5 — accuracy: 0.5789    ksr/// 0.5617

## SHELL_256x256 -> ckpt1/0380.h5 — accuracy: 0.6825    ksr/// 0.6718



## Training & testing \/ \/ \/

## ckpt_shell/0400.h5 — min_loss: 0.0867 (among ckpt_shell)
## ckpt_shell/0400.h5 — max_accuracy: 0.9945 (among ckpt_shell)
## ckpt_shell/0040.h5 — min_val_loss: 0.4115 (among ckpt_shell)
## ckpt_shell/0325.h5 — max_val_accuracy: 0.8465 (among ckpt_shell)
## ckpt_shell/0060.h5 — min_test_loss: 0.0570 (among ckpt_shell, testing on SHELL_256)
## ckpt_shell/0350.h5 — max_test_accuracy: 0.8931 (among ckpt_shell, testing on SHELL_256)

## Testing \/ \/ \/

## S-UNIWARD_256x256_0.1_bpp -> ckpt_shell/0350.h5 — accuracy: 0.5038
## S-UNIWARD_256x256_0.2_bpp -> ckpt_shell/0350.h5 — accuracy: 0.5337
## S-UNIWARD_256x256_0.3_bpp -> ckpt_shell/0350.h5 — accuracy: 0.5928
## S-UNIWARD_256x256_0.4_bpp -> ckpt_shell/0350.h5 — accuracy: 0.6581
## S-UNIWARD_256x256_0.5_bpp -> ckpt_shell/0350.h5 — accuracy: 0.7225

## SHELL_256x256_steghide_Anubis -> ckpt_shell/0350.h5 — accuracy: 0.8149
## SHELL_256x256_steghide_AWFULSHRED -> ckpt_shell/0350.h5 — accuracy: 0.8654
## SHELL_256x256_steghide_DarkRadiation -> ckpt_shell/0350.h5 — accuracy: 0.9083
## SHELL_256x256_steghide_IRCbot -> ckpt_shell/0350.h5 — accuracy: 0.8549

## SHELL_256x256_stegano_Anubis -> ckpt_shell/0350.h5 — accuracy: 0.8901
## SHELL_256x256_stegano_AWFULSHRED -> ckpt_shell/0350.h5 — accuracy: 0.9331
## SHELL_256x256_stegano_DarkRadiation -> ckpt_shell/0350.h5 — accuracy: 0.9427
## SHELL_256x256_stegano_IRCbot -> ckpt_shell/0350.h5 — accuracy: 0.9163

## SHELL_256x256 -> ckpt1/0380.h5 — accuracy: 0.8931
