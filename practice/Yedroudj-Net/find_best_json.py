import json
from pathlib import Path
import sys

def main():
    directory = "S-UNI_256_0.4_yedroudj_net_sgd_cosine_decay_from_1e-2_to_0"
    for function in find_min_loss, find_max_accuracy, find_min_val_loss, find_max_val_accuracy:
        file, value = function(directory)
        if file is None:
            print("Файл не найден")
            sys.exit(1)
        print(f"Метод: {function.__name__}")
        print(f"Файл: {file}")
        print(f"Value: {value}")
        print()



def find_min_loss(dirpath):
    dirp = Path(dirpath)
    if not dirp.is_dir():
        raise NotADirectoryError(f"Not a directory: {dirpath}")

    min_loss = None
    min_file = None

    for p in dirp.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        try:
            loss = data["logs"]["loss"]
        except (TypeError, KeyError):
            continue

        try:
            loss_num = float(loss)
        except (ValueError, TypeError):
            continue

        if min_loss is None or loss_num < min_loss:
            min_loss = loss_num
            min_file = p

    return min_file, min_loss



def find_max_accuracy(dirpath):
    dirp = Path(dirpath)
    if not dirp.is_dir():
        raise NotADirectoryError(f"Not a directory: {dirpath}")

    max_acc = None
    max_file = None

    for p in dirp.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        try:
            acc = data["logs"]["accuracy"]
        except (TypeError, KeyError):
            continue

        try:
            acc_num = float(acc)
        except (ValueError, TypeError):
            continue

        if max_acc is None or acc_num > max_acc:
            max_acc = acc_num
            max_file = p

    return max_file, max_acc



def find_min_val_loss(dirpath):
    dirp = Path(dirpath)
    if not dirp.is_dir():
        raise NotADirectoryError(f"Not a directory: {dirpath}")

    min_val = None
    min_file = None

    for p in dirp.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        try:
            val = data["logs"]["val_loss"]
        except (TypeError, KeyError):
            continue

        try:
            val_num = float(val)
        except (ValueError, TypeError):
            continue

        if min_val is None or val_num < min_val:
            min_val = val_num
            min_file = p

    return min_file, min_val



def find_max_val_accuracy(dirpath):
    dirp = Path(dirpath)
    if not dirp.is_dir():
        raise NotADirectoryError(f"Not a directory: {dirpath}")

    max_val = None
    max_file = None

    for p in dirp.glob("*.json"):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        try:
            val = data["logs"]["val_accuracy"]
        except (TypeError, KeyError):
            continue

        try:
            val_num = float(val)
        except (ValueError, TypeError):
            continue

        if max_val is None or val_num > max_val:
            max_val = val_num
            max_file = p

    return max_file, max_val



if __name__ == "__main__":
    main()
