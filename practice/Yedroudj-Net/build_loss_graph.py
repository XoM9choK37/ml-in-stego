import json
import os
import matplotlib.pyplot as plt

json_directory = 'ckpt_shell'

test_filename = 'SHELL_256_ckpt_shell_testing.txt'

epochs = []
train_losses = []
validation_losses = []

for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        with open(os.path.join(json_directory, filename), 'r') as f:
            data = json.load(f)
            epochs.append(data['epoch'])
            train_losses.append(data['logs']['loss'])
            validation_losses.append(data['logs']['val_loss'])

test_losses = []
test_losses = []

with open(test_filename, 'r') as f:
    try:
        for line in f:
            parts = line.split(maxsplit=1   )
            epoch = int(parts[0])
            values = eval(parts[1])
            test_losses.append(values[0])
            if epoch not in epochs:
                epochs.append(epoch)
    except:
        pass

plt.figure(figsize=(12, 6), tight_layout=True)

plt.plot(epochs, train_losses, marker='', linestyle='-', color='b', label='Фунция потерь обучения')
plt.plot(epochs, validation_losses, marker='', linestyle='-', color='g', label='Фунция потерь валидации')

plt.plot(epochs, test_losses, marker='', linestyle='--', color='r', label='Фунция потерь тестирования')

plt.title('Функции потерь обучения, валидации и тестирования по эпохам')
plt.xlabel('Номер эпохи')
plt.ylabel('Функция потерь')

plt.xticks(epochs, fontsize=9, rotation=75)
plt.xlim(5, 400)
plt.ylim(0, max(max(train_losses), max(validation_losses), max(test_losses)) + 0.01)
plt.grid()
plt.legend(loc='upper left')

plt.show()
