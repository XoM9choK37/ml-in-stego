import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# Путь к папке с изображениями
directory = '../BOSSbase_1.01_bmp'  # Замените это на ваш путь

# Считываем изображения из папки
images = []
for filename in os.listdir(directory):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.pgm', '.bmp')) and len(images) < 10:
        img = mpimg.imread(os.path.join(directory, filename))
        images.append(img)

# Проверяем, что изображений достаточно
if len(images) < 10:
    raise ValueError("В папке должно быть как минимум 10 изображений.")

# Создаем фигуру и оси
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Отображаем изображения
for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i])
    ax.axis('off')  # Убираем оси

plt.tight_layout()
plt.show()
