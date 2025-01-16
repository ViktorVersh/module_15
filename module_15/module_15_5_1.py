# Импортируем необходимые библиотеки
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Загружаем датасет Iris
wine = load_wine()
X = wine.data
y = wine.target

# Разделяем данные на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучаем модель логистической регрессии
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Предсказываем результаты на тестовых данных
y_pred = model.predict(X_test)

# Оцениваем качество модели
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))