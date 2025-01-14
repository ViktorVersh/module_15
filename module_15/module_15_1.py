from sklearn.neural_network import MLPClassifier
import numpy as np

# Данные для обучения (XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # Операция XOR

# Создание и обучение MLP-классификатора
mlp = MLPClassifier(hidden_layer_sizes=(5,), activation='relu', max_iter=2500, solver='lbfgs')
mlp.fit(X, y)

# Тестирование
for xi in X:
    print(f"{xi} -> {mlp.predict([xi])[0]}")
