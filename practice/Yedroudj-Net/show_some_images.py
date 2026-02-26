import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

directory = 'D:/Documents/DATASETS/BOSSbase_1.01_bmp'

images = []
for filename in os.listdir(directory):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.pgm', '.bmp')) and len(images) < 10:
        img = mpimg.imread(os.path.join(directory, filename))
        images.append(img)

if len(images) < 10:
    raise ValueError("В папке должно быть как минимум 10 изображений.")

fig, axes = plt.subplots(2, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i])
    ax.axis('off')

plt.tight_layout()
plt.show()
