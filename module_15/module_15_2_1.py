from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Генерация синтетических данных
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_classes=3)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Создание модели многослойного перцептрона
mlp = MLPClassifier(hidden_layer_sizes=(50, 30, 10),
                    activation='relu',
                    solver='adam',
                    max_iter=500,
                    learning_rate_init=0.01,
                    random_state=42)

# Обучение модели
mlp.fit(X_train, y_train)

# Оценка точности модели на тестовых данных
accuracy = mlp.score(X_test, y_test)
print(f'Точность модели: {accuracy:.4f}')
