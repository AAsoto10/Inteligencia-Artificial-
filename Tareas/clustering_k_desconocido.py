import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Carga del dataset real Mall Customers
df = pd.read_csv("Mall_Customers.csv")

print("Primeras filas del dataset:")
print(df.head())
print(f"\nDimensiones: {df.shape[0]} clientes, {df.shape[1]} columnas")
print(f"\nColumnas: {list(df.columns)}")
print("\nEstadisticas descriptivas:")
print(df.describe())


# Seleccionamos las 2 columnas numericas para el clustering
X_data = df[["Annual Income (k$)", "Spending Score (1-100)"]].values

print("Shape de X:", X_data.shape)
print("Primeros 5 valores de X:\n", X_data[:5])


def plot_clusters(data, y=None):
    plt.scatter(data[:, 0], data[:, 1], c=y, s=10)
    plt.xlabel("Ingresos anuales (k$)", fontsize=14)
    plt.ylabel("Puntaje de Gasto", fontsize=14, rotation=90)


plt.figure(figsize=(8, 4))
plot_clusters(X_data)
plt.title("Clientes del Mall (sin etiquetas)")
plt.show()


# =============================
# AQUI CAMBIA: k desconocido
# =============================
K = range(2, 11)
wcss = []
sil_scores = []

for k_try in K:
    km = KMeans(n_clusters=k_try, random_state=42, n_init=10)
    yk = km.fit_predict(X_data)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_data, yk))


def detectar_codo_por_distancia(ks, inercias):
    p1 = np.array([ks[0], inercias[0]], dtype=float)
    p2 = np.array([ks[-1], inercias[-1]], dtype=float)
    distancias = []
    for k_val, iner in zip(ks, inercias):
        p = np.array([k_val, iner], dtype=float)
        num = np.abs(np.cross(p2 - p1, p1 - p))
        den = np.linalg.norm(p2 - p1)
        distancias.append(num / den)
    return ks[int(np.argmax(distancias))]


k_codo = detectar_codo_por_distancia(list(K), wcss)
k_sil = list(K)[int(np.argmax(sil_scores))]
k_final = int(round((k_codo + k_sil) / 2))

print("\nSeleccion automatica de k")
print("k por codo:", k_codo)
print("k por silhouette:", k_sil)
print("k final recomendado:", k_final)


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(list(K), wcss, marker="o")
plt.axvline(k_codo, color="red", linestyle="--", label=f"k_codo={k_codo}")
plt.axvline(k_final, color="green", linestyle=":", label=f"k_final={k_final}")
plt.title("Metodo del codo")
plt.xlabel("Numero de clusters (k)")
plt.ylabel("WCSS")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(list(K), sil_scores, marker="o")
plt.axvline(k_sil, color="red", linestyle="--", label=f"k_sil={k_sil}")
plt.axvline(k_final, color="green", linestyle=":", label=f"k_final={k_final}")
plt.title("Silhouette score")
plt.xlabel("Numero de clusters (k)")
plt.ylabel("Silhouette")
plt.legend()

plt.tight_layout()
plt.show()


kmeans = KMeans(n_clusters=k_final, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X_data)
print("\nEtiquetas de cluster (primeros 20):")
print(y_pred[:20])


# Centroides encontrados por K-Means
print("\nCentroides:")
print(kmeans.cluster_centers_)


# Prediccion de nuevos puntos
X_new = np.array([[50, 50], [100, 80], [20, 20], [80, 10]])
print("\nCluster asignado a nuevos puntos:")
print(kmeans.predict(X_new))


def plot_data(data):
    plt.plot(data[:, 0], data[:, 1], "k.", markersize=4)


def plot_centroids(centroids, weights=None, circle_color="w", cross_color="k"):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="o",
        s=10,
        linewidths=8,
        color=circle_color,
        zorder=10,
        alpha=0.9,
    )
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="x",
        s=2,
        linewidths=10,
        color=cross_color,
        zorder=11,
        alpha=1,
    )


def plot_decision_boundaries(
    clusterer,
    data,
    resolution=1000,
    show_centroids=True,
    show_xlabels=True,
    show_ylabels=True,
):
    mins = data.min(axis=0) - 0.1
    maxs = data.max(axis=0) + 0.1
    xx, yy = np.meshgrid(
        np.linspace(mins[0], maxs[0], resolution),
        np.linspace(mins[1], maxs[1], resolution),
    )
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel2")
    plt.contour(
        Z,
        extent=(mins[0], maxs[0], mins[1], maxs[1]),
        linewidths=1,
        colors="k",
    )
    plot_data(data)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)

    if show_xlabels:
        plt.xlabel("Annual Income (k$)", fontsize=14)
    else:
        plt.tick_params(labelbottom=False)
    if show_ylabels:
        plt.ylabel("Spending Score", fontsize=14, rotation=90)
    else:
        plt.tick_params(labelleft=False)


plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X_data)
plt.title("K-Means - Fronteras de decision (Mall Customers)", fontsize=14)
plt.show()
