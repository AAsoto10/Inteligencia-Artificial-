import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def train(episodes):
    env = gym.make('MountainCar-v0')

    # --- Discretización del espacio de observación continuo ---
    # MountainCar tiene 2 variables continuas: posición y velocidad
    # Las dividimos en "bins" para poder usar una tabla Q discreta
    pos_bins = 20       # número de divisiones para la posición [-1.2, 0.6]
    vel_bins = 20       # número de divisiones para la velocidad [-0.07, 0.07]
    n_actions = env.action_space.n  # 3 acciones: izquierda, nada, derecha

    # Tabla Q: dimensiones [pos_bins x vel_bins x n_actions]
    q_table = np.zeros((pos_bins, vel_bins, n_actions))

    # Límites del espacio de observación
    pos_low, vel_low = env.observation_space.low
    pos_high, vel_high = env.observation_space.high

    # --- Hiperparámetros ---
    learning_rate = 0.1
    discount_factor = 0.95
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    min_epsilon = 0.01
    rng = np.random.default_rng()

    rewards_per_episode = np.zeros(episodes)

    def discretize(obs):
        """Convierte observación continua a índices discretos para la tabla Q."""
        pos, vel = obs
        pos_idx = int(np.digitize(pos, np.linspace(pos_low, pos_high, pos_bins - 1)))
        vel_idx = int(np.digitize(vel, np.linspace(vel_low, vel_high, vel_bins - 1)))
        pos_idx = np.clip(pos_idx, 0, pos_bins - 1)
        vel_idx = np.clip(vel_idx, 0, vel_bins - 1)
        return pos_idx, vel_idx


    for i in range(episodes):
        # Solo renderizar en los últimos 10 episodios
        render = (i >= episodes - 10)
        if render and env.render_mode != 'human':
            env.close()
            env = gym.make('MountainCar-v0', render_mode='human')
        elif not render and env.render_mode == 'human':
            env.close()
            env = gym.make('MountainCar-v0')

        obs, _ = env.reset()
        state = discretize(obs)

        terminated = False
        truncated = False
        total_reward = 0

        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state[0], state[1], :])

            new_obs, reward, terminated, truncated, _ = env.step(action)
            new_state = discretize(new_obs)

            position, velocity = new_obs
            custom_reward = reward + 10 * (abs(velocity) + 0.1 * position)

            q_table[state[0], state[1], action] = (
                q_table[state[0], state[1], action]
                + learning_rate * (
                    custom_reward
                    + discount_factor * np.max(q_table[new_state[0], new_state[1], :])
                    - q_table[state[0], state[1], action]
                )
            )

            state = new_state
            total_reward += reward

        epsilon = max(epsilon - epsilon_decay_rate, min_epsilon)

        if terminated and not truncated:
            rewards_per_episode[i] = 1

        if (i + 1) % 500 == 0:
            recent_success = np.sum(rewards_per_episode[max(0, i - 499):i + 1])
            print(f'Episodio {i + 1:5d} | Recompensa total: {total_reward:7.1f} | '
                  f'Épsilon: {epsilon:.4f} | Éxitos últimos 500: {recent_success:.0f}')

    env.close()

    print('\nTabla Q final (muestra de estados centrales):')
    print(q_table[10, :, :])

    # Gráfica de éxitos acumulados en ventanas de 100 episodios
    sum_rewards = np.array([
        np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
        for t in range(episodes)
    ])

    plt.figure(figsize=(10, 5))
    plt.plot(sum_rewards, color='steelblue', linewidth=1.2)
    plt.title('Mountain Car — Éxitos acumulados (ventana de 100 episodios)')
    plt.xlabel('Episodio')
    plt.ylabel('Éxitos en últimos 100 episodios')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    train(10000)
