# # import cv2
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Загрузка изображения
# image = cv2.imread('/papil.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Применение оператора Собеля для вычисления градиентов по осям X и Y
# sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)  # Градиент по оси X
# sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)  # Градиент по оси Y
#
# # Вычисление величины градиента
# gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
#
# # Нормализация градиентов для отображения
# sobel_x = cv2.convertScaleAbs(sobel_x)
# sobel_y = cv2.convertScaleAbs(sobel_y)
# gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
#
# # Отображение результатов
# plt.figure(figsize=(15, 5))
#
# plt.subplot(1, 4, 1)
# plt.title('Оригинальное изображение')
# plt.imshow(image, cmap='gray')
#
# plt.subplot(1, 4, 2)
# plt.title('Градиент по оси X')
# plt.imshow(sobel_x, cmap='gray')
#
# plt.subplot(1, 4, 3)
# plt.title('Градиент по оси Y')
# plt.imshow(sobel_y, cmap='gray')
#
# plt.subplot(1, 4, 4)
# plt.title('Величина градиента')
# plt.imshow(gradient_magnitude, cmap='gray')
#
# plt.show()


# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Загрузка изображения
# image = cv2.imread('/papil.jpg', cv2.IMREAD_GRAYSCALE)
#
# # Применение гауссового фильтра для снижения шумов
# blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)
#
# # Применение Канни Эдж Детектора
# edges = cv2.Canny(blurred_image, 50, 150)
# # 50 и 150 - это  high threshold и low threshold —
# # два пороговых значения, которые используются
# # для определения сильных и слабых границ на изображении.
#
# # Отображение результата
# plt.figure(figsize=(10, 5))
#
# plt.subplot(1, 2, 1)
# plt.title('Оригинальное изображение')
# plt.imshow(image, cmap='gray')
#
# plt.subplot(1, 2, 2)
# plt.title('Обнаруженные границы')
# plt.imshow(edges, cmap='gray')
#
# plt.show()


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image = cv2.imread('/papil.jpg', cv2.IMREAD_GRAYSCALE)

# Проверка успешности загрузки изображения
if image is None:
    print("Ошибка: изображение не найдено.")
    exit()

# Применение гауссового фильтра для снижения шумов
blurred_image = cv2.GaussianBlur(image, (5, 5), 1.4)

# Применение Канни Эдж Детектора
edges = cv2.Canny(blurred_image, 50, 150)

# Отображение результата
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Оригинальное изображение')
plt.imshow(image, cmap='gray')
plt.axis('off')  # Убираем оси

plt.subplot(1, 2, 2)
plt.title('Обнаруженные границы')
plt.imshow(edges, cmap='gray')
plt.axis('off')  # Убираем оси

plt.tight_layout()  # Убираем лишние отступы между графиками
plt.show()