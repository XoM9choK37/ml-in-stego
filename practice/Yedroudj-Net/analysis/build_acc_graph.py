import json
import os
import matplotlib.pyplot as plt

# Директория с JSON-файлами
json_directory = 'ckpt_shell'

# Путь к файлу с данными тестирования
test_filename = 'SHELL_256_ckpt_shell_testing.txt'

# Списки для хранения данных
epochs = []
train_accuracies = []
validation_accuracies = []

# Чтение JSON-файлов из директории
for filename in os.listdir(json_directory):
    if filename.endswith('.json'):
        with open(os.path.join(json_directory, filename), 'r') as f:
            data = json.load(f)
            epochs.append(data['epoch'])
            train_accuracies.append(data['logs']['accuracy'])
            validation_accuracies.append(data['logs']['val_accuracy'])

# Списки для тестовых данных
test_losses = []
test_accuracies = []

# Чтение тестового файла и извлечение данных
with open(test_filename, 'r') as f:
    try:
        for line in f:
            parts = line.split(maxsplit=1   )  # Разделение строки на части
            epoch = int(parts[0])  # Первая часть — номер эпохи
            values = eval(parts[1])  # Вторая часть — список значений
            test_accuracies.append(values[1])  # Второе значение — test_accuracy
            # Добавляем позицию, чтобы соответствовать эпохам
            if epoch not in epochs:
                epochs.append(epoch)
    except:
        pass

# Создание графика
plt.figure(figsize=(12, 6), tight_layout=True)

# Построение графиков для train и validation accuracy
plt.plot(epochs, train_accuracies, marker='', linestyle='-', color='b', label='Точность обучения')
plt.plot(epochs, validation_accuracies, marker='', linestyle='-', color='g', label='Точность валидации')

# Построение графика для test_accuracy
plt.plot(epochs, test_accuracies, marker='', linestyle='--', color='r', label='Точность тестирования')

# Настройки графика
plt.title('Точности обучения, валидации и тестирования по эпохам')
plt.xlabel('Номер эпохи')
plt.ylabel('Точность')

# Установка меток по оси X с вертикальным текстом
plt.xticks(epochs, fontsize=9, rotation=75)  # Вертикальная ориентация меток
plt.xlim(5, 400)
plt.ylim(0, 1)  # Ограничение по оси Y от 0 до 1 для точностей
plt.grid()
plt.legend(loc='upper left')  # Добавление легенды для обозначения линий

# Отображение графика
plt.show()
