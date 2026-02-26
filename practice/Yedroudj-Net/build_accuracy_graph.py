import json
import os
import matplotlib.pyplot as plt

json_directory = 'ckpt_shell'

test_filename = 'SHELL_256_ckpt_shell_testing.txt'

epochs = []
train_accuracies = []
validation_accuracies = []

for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        with open(os.path.join(json_directory, filename), 'r') as f:
            data = json.load(f)
            epochs.append(data['epoch'])
            train_accuracies.append(data['logs']['accuracy'])
            validation_accuracies.append(data['logs']['val_accuracy'])

test_losses = []
test_accuracies = []

with open(test_filename, 'r') as f:
    try:
        for line in f:
            parts = line.split(maxsplit=1   )
            epoch = int(parts[0])
            values = eval(parts[1])
            test_accuracies.append(values[1])
            if epoch not in epochs:
                epochs.append(epoch)
    except:
        pass

plt.figure(figsize=(12, 6), tight_layout=True)

plt.plot(epochs, train_accuracies, marker='', linestyle='-', color='b', label='Точность обучения')
plt.plot(epochs, validation_accuracies, marker='', linestyle='-', color='g', label='Точность валидации')

plt.plot(epochs, test_accuracies, marker='', linestyle='--', color='r', label='Точность тестирования')

plt.title('Точности обучения, валидации и тестирования по эпохам')
plt.xlabel('Номер эпохи')
plt.ylabel('Точность')

plt.xticks(epochs, fontsize=9, rotation=75)
plt.xlim(5, 400)
plt.ylim(0, 1)
plt.grid()
plt.legend(loc='upper left')

plt.show()
