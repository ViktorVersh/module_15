import numpy as np


# Функция активации (шаговая функция)
def step_function(x):
    """
    Шаговая функция активации
    :param x:
    :return:
    """
    # return np.where(x >= 0, 1, 0)
    return 1 if x >= 0 else 0


class Perceptron:
    """
    Персептрон
    """

    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        self.W = np.zeros(input_size + 1)  # вектор весов
        self.learning_rate = learning_rate  # скорость обучения
        self.epochs = epochs  # количество циклов обучения

    def predict(self, x):
        """
        Предсказание на основе текущих весов и входящего вектора 'x'
        :param x:
        :return:
        """
        return step_function(np.dot(self.W, x))

    def train(self, X, y):
        """
        Обучение персептрона
        :param X:
        :param y:
        :return:
        """
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)  # Вставка смещения (bias)
                prediction = self.predict(xi)  # Предсказание
                self.W += self.learning_rate * (target - prediction) * xi  # циклический сдвиг весов, веса обновляются
                # согласно правила Delta Rule: W=W+learning_rate*(target-prediction)*xi


# Данные для обучения (И, ИЛИ)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Операция И (AND)

perceptron = Perceptron(input_size=2)
perceptron.train(X, y)

# Тестирование
for xi in X:
    xi_with_bias = np.insert(xi, 0, 1)  # Вставка смещения (bias) для тестирования
    print(f"{xi} -> {perceptron.predict(xi_with_bias)}")
