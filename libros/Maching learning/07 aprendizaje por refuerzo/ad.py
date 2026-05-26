import numpy as np

np.random.seed(42)
n = 16
recompensaDado = np.zeros(n)
recompensaDado[2]  =  1.0
recompensaDado[7]  =  1.0
recompensaDado[13] =  1.0
recompensaDado[5]  = -1.0
recompensaDado[11] = -1.0
mejor_casilla = int(np.argmax(recompensaDado))

partidas = 1000
turnos = n
alpha = 0.5
epsilon = 0.1  # puedes cambiarlo a 0.0 para greedy puro
recompensas_medias = np.zeros(turnos)
acciones_optimas = np.zeros(turnos)

for partida in range(partidas):
    Q = {k: 0 for k in range(n)}
    for t in range(turnos):
        # Selección de acción (epsilon-greedy)
        if np.random.uniform(0,1) < epsilon:
            a = np.random.randint(n)
        else:
            maxQ = max(Q.values())
            mejores = [k for k, v in Q.items() if v == maxQ]
            a = np.random.choice(mejores)
        recompensa = recompensaDado[a] + np.random.normal(0, 0.3)
        Q[a] += alpha * (recompensa - Q[a])
        recompensas_medias[t] += recompensa
        acciones_optimas[t] += (a == mejor_casilla)

recompensas_medias /= partidas
acciones_optimas /= partidas

Q_final = np.array([Q[k] for k in range(n)])
print(Q_final.reshape(4, 4))
print(f"La casilla con mayor valor aprendido: {int(np.argmax(Q_final))}")
print(f"La casilla con recompensa real más alta: {mejor_casilla}")