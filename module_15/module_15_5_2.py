# from sklearn import datasets
# from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
#
# # Загружаем датасет digits
# digits = datasets.load_digits()
# X = digits.data
# y = digits.target
#
# # Применяем алгоритм DBSCAN
# dbscan = DBSCAN(eps=4, min_samples=20)
# labels = dbscan.fit_predict(X)
#
# # Количество кластеров
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print(f'Количество кластеров: {n_clusters_}')
#
# # Визуализируем результаты
# plt.figure(figsize=(8, 6))
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
# for i in range(len(labels)):
#     plt.text(X[i, 0], X[i, 1], str(y[i]),
#              color=colors[labels[i] % len(colors)],
#              fontdict={'weight': 'bold', 'size': 9})
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.title("DBSCAN Clustering on Digits Dataset")
# plt.show()
#
# import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
#
# # Чтение данных из файла CSV
# wine_data = pd.read_csv('wine.csv')
#
# # Выбор признаков для кластеризации
# features = wine_data[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols']]
#
# # Масштабирование данных
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)
#
# # Применение алгоритма K-средних
# kmeans = KMeans(n_clusters=3, random_state=0)
# clusters = kmeans.fit_predict(scaled_features)
#
# # Добавление столбца с метками кластеров в исходный DataFrame
# wine_data['cluster'] = clusters
#
# # Визуализация результатов кластеризации
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.scatter(wine_data['Alcohol'], wine_data['Malic acid'], c=wine_data['cluster'])
# ax.set_xlabel('Alcohol')
# ax.set_ylabel('Malic acid')
# ax.set_title('Кластеризация вина методом K-средних')
# plt.show()
#
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
#
#
#
# # Разделим данные на признаки и целевые значения
# X = data[:, :-1]
# y = data[:, -1].astype(int)
#
# # Масштабирование данных
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# # Применение алгоритма K-средних
# kmeans = KMeans(n_clusters=3, random_state=0)
# clusters = kmeans.fit_predict(X_scaled)
#
# # Визуализация результатов кластеризации
# plt.figure(figsize=(8, 6))
# plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
# plt.xlabel('Первый признак')
# plt.ylabel('Второй признак')
# plt.title('Кластеризация вина методом K-средних')
# plt.show()

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Загрузить датасет Wine
wine = datasets.load_wine()
X = wine.data
y = wine.target

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Применение алгоритма K-средних
kmeans = KMeans(n_clusters=3, random_state=0)
clusters = kmeans.fit_predict(X_scaled)

# Визуализация результатов кластеризации
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.xlabel('Первый признак')
plt.ylabel('Второй признак')
plt.title('Кластеризация вина методом K-средних')
plt.show()
