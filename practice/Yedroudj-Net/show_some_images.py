import matplotlib.pyplot as plt
from PIL import Image
import os

directory = 'D:/Documents/DATASETS/BOSSbase_1.01_bmp/cover_images'
number_of_images = 20

images = []
for filename in os.listdir(directory):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.pgm', '.bmp')) and len(images) < number_of_images:
        image = Image.open(os.path.join(directory, filename))
        images.append(image)
        continue
    break

fig, axes = plt.subplots(number_of_images // 5, 5, figsize=(15, 6))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(images[i])
    ax.axis('off')

plt.tight_layout()
plt.show()
