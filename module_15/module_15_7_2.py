"""
Это реализация кода с использованием алгоритма SARSA

Основные различия между Q-Learning и SARSA:
1. Выбор следующего действия: В Q-Learning следующее действие выбирается после обновления Q-значения,
   тогда как в SARSA следующее действие выбирается перед обновлением.
2. Формула обновления: В Q-Learning используется максимальное ожидаемое значение следующего состояния,
   а в SARSA — реальное действие, которое было выбрано для следующего состояния.

В остальном код практически идентичен версии с Q-Learning.
"""
import numpy as np
import matplotlib.pyplot as plt

# Параметры
alpha = 0.1  # Скорость обучения
gamma = 0.9  # Коэффициент дисконта (важность будущих вознаграждений относительно текущих)
epsilon = 0.1  # Вероятность случайного выбора
episodes = 1000  # Цикл обучения - количество итераций
grid_size = 5  # размер сетки, гле перемещается агент

# Действия: вверх, вниз, влево, вправо
"""
Действия представлены в виде смещений координат (x, y):
- (-1, 0) — движение вверх,
- (1, 0) — движение вниз,
- (0, -1) — движение влево,
- (0, 1) — движение вправо.
"""
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Инициализация Q-таблицы
"""
Создание матрицы Q где каждая ячейка представляет собой оценку качества действия в данной клетке. 
Изначально все оценки равны нулю.
"""
Q = np.zeros((grid_size, grid_size, len(actions)))


# Функции для выбора действий и обновления Q-таблицы
def choose_action(state):
    """
    Выбор действия для текущего состояния если сгенерированное число меньше epsilon,
    выбирается случайное действие,
    иначе выбирается действие с наибольшей оценкой качества
    :param state:
    :return:
    """
    if np.random.rand() < epsilon:
        return np.random.randint(len(actions))  # Случайное действие
    else:
        return np.argmax(Q[state])  # Лучшее известное действие


def step(state, action):
    """
    Выполнение действия и переход в новое состояние
    :param state:
    :param action:
    :return:
    """
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])

    # Проверяем выход за границу
    if next_state[0] < 0 or next_state[0] >= grid_size or next_state[1] < 0 or next_state[1] >= grid_size:
        next_state = state  # Остаемся на месте

    reward = 1 if next_state == (grid_size - 1, grid_size - 1) else -0.1
    done = next_state == (grid_size - 1, grid_size - 1)

    return next_state, reward, done


def update_q(state, action, reward, next_state, next_action):
    """
    Обновление Q-таблицы на основе временного разностного сигнала (td_error)
    :param state:
    :param action:
    :param reward:
    :param next_state:
    :param next_action:
    :return:
    """
    td_target = reward + gamma * Q[next_state][next_action]
    td_error = td_target - Q[state][action]
    Q[state][action] += alpha * td_error


# Обучение агента
for episode in range(episodes):
    state = (0, 0)
    done = False
    action = choose_action(state)

    while not done:
        next_state, reward, done = step(state, action)
        next_action = choose_action(next_state)
        update_q(state, action, reward, next_state, next_action)
        state = next_state
        action = next_action

# Визуализация политики
policy = np.argmax(Q, axis=2)
print("Оптимальная политика:")
print(policy)

# Функция ценности
value_function = np.max(Q, axis=2)

plt.figure(figsize=(10, 6))

# Политика
plt.subplot(1, 2, 1)
plt.title('Политика')
plt.imshow(policy, cmap='viridis', origin='upper')
for i in range(grid_size):
    for j in range(grid_size):
        plt.text(j, i, policy[i, j], ha='center', va='center', color='white')

# Функция ценности
plt.subplot(1, 2, 2)
plt.title('Функция ценности')
plt.imshow(value_function, cmap='viridis', origin='upper')
for i in range(grid_size):
    for j in range(grid_size):
        plt.text(j, i, round(value_function[i, j], 2), ha='center', va='center', color='white')

plt.show()
