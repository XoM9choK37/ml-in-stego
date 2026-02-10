import json
import os
import matplotlib.pyplot as plt

# Директория с JSON-файлами
json_directory = 'ckpt_shell'

# Путь к файлу с данными тестирования
test_filename = 'SHELL_256_ckpt_shell_testing.txt'

# Списки для хранения данных
epochs = []
train_losses = []
validation_losses = []

# Чтение JSON-файлов из директории
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        with open(os.path.join(json_directory, filename), 'r') as f:
            data = json.load(f)
            epochs.append(data['epoch'])
            train_losses.append(data['logs']['loss'])
            validation_losses.append(data['logs']['val_loss'])

# Списки для тестовых данных
test_losses = []
test_losses = []

# Чтение тестового файла и извлечение данных
with open(test_filename, 'r') as f:
    try:
        for line in f:
            parts = line.split(maxsplit=1   )  # Разделение строки на части
            epoch = int(parts[0])  # Первая часть — номер эпохи
            values = eval(parts[1])  # Вторая часть — список значений
            test_losses.append(values[0])  # Второе значение — test_loss
            # Добавляем позицию, чтобы соответствовать эпохам
            if epoch not in epochs:
                epochs.append(epoch)
    except:
        pass

# Создание графика
plt.figure(figsize=(12, 6), tight_layout=True)

# Построение графиков для train и validation loss
plt.plot(epochs, train_losses, marker='', linestyle='-', color='b', label='Фунция потерь обучения')
plt.plot(epochs, validation_losses, marker='', linestyle='-', color='g', label='Фунция потерь валидации')

# Построение графика для test_loss
plt.plot(epochs, test_losses, marker='', linestyle='--', color='r', label='Фунция потерь тестирования')

# Настройки графика
plt.title('Функции потерь обучения, валидации и тестирования по эпохам')
plt.xlabel('Номер эпохи')
plt.ylabel('Функция потерь')

# Установка меток по оси X с вертикальным текстом
plt.xticks(epochs, fontsize=9, rotation=75)  # Вертикальная ориентация меток
plt.xlim(5, 400)
plt.ylim(0, max(max(train_losses), max(validation_losses), max(test_losses)) + 0.01)
plt.grid()
plt.legend(loc='upper left')  # Добавление легенды для обозначения линий

# Отображение графика
plt.show()
