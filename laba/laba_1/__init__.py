import matplotlib

matplotlib.use('TkAgg')
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('image.jpg')

blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

kernel = np.array([[0, -1, 0],
                   [-1, 5, -1],
                   [0, -1, 0]])
sharpened = cv2.filter2D(image, -1, kernel)

blurred = cv2.GaussianBlur(image, (5, 5), 0)
sharpened_mask = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

edges_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
edges_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
edges = cv2.sqrt(edges_x ** 2 + edges_y ** 2)
edges = cv2.convertScaleAbs(edges)

combined = cv2.addWeighted(blurred, 0.5, edges, 0.5, 0)
combined = cv2.addWeighted(combined, 0.5, sharpened, 0.5, 0)


def show_images(original, blurred, edges, sharpened, combined):
    plt.figure(figsize=(12, 10))

    plt.subplot(2, 3, 1)
    plt.title('Оригинальное изображение')
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Размытие по Гауссу')
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title('Выделение границ')
    plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title('Повышение резкости (метод 1)')
    plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Повышение резкости (метод 2)')
    plt.imshow(cv2.cvtColor(sharpened_mask, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title('Комбинация изображений')
    plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

show_images(image, blurred_image, edges, sharpened, combined)
