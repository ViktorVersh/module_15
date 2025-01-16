# Импортируем необходимые библиотеки
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Загрузим датасет
wine = load_wine() #  wine - датасет Вина из винограда трех сортов
X = wine.data

# Масштабируем данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Обучаем модель K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Предсказываем кластеры
clusters = kmeans.predict(X_scaled)

# Визуализируем результаты кластеризации
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 6], c=clusters, cmap='viridis', marker='o', s=80, edgecolor='black')
plt.title('Кластеризация вина методом K-means')
plt.xlabel('Содержание алкоголя')
plt.ylabel('Концентрация флавоноидов')
plt.grid(True)  # Выводит сетку на графике
plt.show()  # Выводит график на экран
