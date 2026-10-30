import numpy as np
import matplotlib.pyplot as plt

def calculer_vitesses(potentiel, domaine, dx, dy):
    u = np.zeros_like(potentiel)
    v = np.zeros_like(potentiel)

    for i in range(1, potentiel.shape[0]-1):
        for j in range(1, potentiel.shape[1]-1):
            if domaine[i, j] == 1 or domaine[i, j] == 2:
                if domaine[i, j-1] != 0:
                    u[i, j] = -(potentiel[i, j] - potentiel[i, j-1]) / dx

    for i in range(1, potentiel.shape[0]-1):
        for j in range(1,potentiel.shape[1]-1):
            if domaine[i, j] == 1 or domaine[i, j] == 2:
                if domaine[i+1, j] != 0:
                    v[i, j] = (potentiel[i, j] - potentiel[i+1, j]) / dy

    return u, v


def afficher_champs_vitesse(u, v, domaine):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ny, nx = u.shape

    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    pas = 10

    # Facteur d'amplification visuelle des flèches
    amplification = 20

    # --------------------------------------------------
    # Champ de vitesse horizontal u
    # --------------------------------------------------
    Xu = X - 0.5
    Yu = Y

    axes[0].imshow(domaine, origin='upper')

    axes[0].quiver(
        Xu[::pas, ::pas],
        Yu[::pas, ::pas],
        amplification * u[::pas, ::pas],
        np.zeros_like(u[::pas, ::pas]),
        color='red',
        angles='xy',
        scale_units='xy',
        scale=1,
        width=0.003,
        headwidth=5,
        headlength=7
    )

    axes[0].set_xticks(np.arange(-0.5, nx, 1), minor=True)
    axes[0].set_yticks(np.arange(-0.5, ny, 1), minor=True)
    axes[0].grid(
        which='minor',
        linestyle='-',
        linewidth=0.5,
        color='black',
        alpha=0.5
    )

    axes[0].set_title("Champ de la composante de vitesse horizontale u")
    axes[0].set_xlabel("indices colonnes")
    axes[0].set_ylabel("indices lignes")

    # --------------------------------------------------
    # Champ de vitesse vertical v
    # --------------------------------------------------
    Xv = X
    Yv = Y + 0.5

    axes[1].imshow(domaine, origin='upper')

    axes[1].quiver(
        Xv[::pas, ::pas],
        Yv[::pas, ::pas],
        np.zeros_like(v[::pas, ::pas]),
        amplification * v[::pas, ::pas],
        color='blue',
        angles='xy',
        scale_units='xy',
        scale=1,
        width=0.003,
        headwidth=5,
        headlength=7
    )

    axes[1].set_xticks(np.arange(-0.5, nx, 1), minor=True)
    axes[1].set_yticks(np.arange(-0.5, ny, 1), minor=True)
    axes[1].grid(
        which='minor',
        linestyle='-',
        linewidth=0.5,
        color='black',
        alpha=0.5
    )

    axes[1].set_title("Champ de la composante de vitesse verticale v")
    axes[1].set_xlabel("indices colonnes")
    axes[1].set_ylabel("indices lignes")

    plt.tight_layout()
    plt.savefig("champs_vitesse.png", dpi=300, bbox_inches="tight")
    plt.show()