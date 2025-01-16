from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Загружаем датасет
digits = load_digits()
X = digits.data
y = digits.target

# Разделяем данные на обучающую и тестовую части
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = SVC(gamma='scale', max_iter=10000)
clf.fit(X_train, y_train)

# Прогнозирование результатов на тестовом наборе
y_pred = clf.predict(X_test)

# Оценка точности модели
accuracy = accuracy_score(y_test, y_pred)
print(f'Точность модели: {accuracy:.2f}')

# Визуализация нескольких примеров
plt.figure(figsize=(12, 3))
for index in range(10):
    plt.subplot(1, 10, index + 1)
    plt.imshow(X_test[index].reshape((8, 8)), cmap="Grays")
    plt.title(f'Число:\n {y_pred[index]}')
plt.show()
