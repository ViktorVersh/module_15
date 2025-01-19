"""
Это реализация кода с использованием алгоритма Q-Learning
"""
import numpy as np
import matplotlib.pyplot as plt

# Параметры
alpha = 0.1  # Скорость обучения
gamma = 0.9  # Коэффициент дисконта (важность будущих вознаграждений относительно текущих)
epsilon = 0.1  # Вероятность случайного выбора
episodes = 1000  # Количество итераций - эпизодов обучения
grid_size = 5  # размер сетки, где перемещается агент


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
Q = np.zeros((grid_size, grid_size, 4))


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
        return np.random.randint(4)
    else:
        return np.argmax(Q[state])


def step(state, action):
    """
    Функция вычисляет следующее состояние на основании текущего состояния и выбранного действия,
    если новое состояние выходит за границы сетки, оно остается прежним. Вознаграждение reward = 1 если достигнута
    конечная клетка, иначе 0.1
    :param state:
    :param action:
    :return:
    """
    next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
    if next_state[0] < 0 or next_state[0] >= grid_size or next_state[1] < 0 or next_state[1] >= grid_size:
        next_state = state  # оставаться на месте, если выходит за пределы
    reward = 1 if next_state == (grid_size-1, grid_size-1) else -0.1
    done = next_state == (grid_size-1, grid_size-1)
    return next_state, reward, done


def update_q(state, action, reward, next_state):
    """
    Обновление Q-таблицы на уравнения Бэллмана:
    - Выбирается лучшее возможное действие в следующем состоянии.
    - Рассчитывается целевая оценка (td_target) как сумма текущего вознаграждения и ожидаемого
    будущего вознаграждения, умноженного на `gamma`.
    - Вычисляется ошибка временного различия (td_error) как разница между целевой оценкой и текущей оценкой действия.
    - Значение в Q-таблице обновляется на величину ошибки, умноженную на alpha.
    :param state:
    :param action:
    :param reward:
    :param next_state:
    :return:
    """
    best_next_action = np.argmax(Q[next_state])
    td_target = reward + gamma * Q[next_state][best_next_action]
    td_error = td_target - Q[state][action]
    Q[state][action] += alpha * td_error


# Обучение агента
for episode in range(episodes):
    state = (0, 0)
    done = False
    while not done:
        action = choose_action(state)
        next_state, reward, done = step(state, action)
        update_q(state, action, reward, next_state)
        state = next_state

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
