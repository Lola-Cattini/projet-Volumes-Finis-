import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

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

    ny, nx = u.shape

    # Coordonnées des centres des mailles
    X, Y = np.meshgrid(np.arange(nx), np.arange(ny))

    # Pas d'affichage des flèches
    pas = 10

    # Seuil pour éliminer les vitesses trop faibles
    seuil_u = 0.02 * np.max(np.abs(u))
    seuil_v = 0.02 * np.max(np.abs(v))

    # Sélection des mailles fluides
    fluide = domaine != 0

    masque_u = fluide & (np.abs(u) > seuil_u)
    masque_v = fluide & (np.abs(v) > seuil_v)

    # Normalisation des longueurs pour améliorer la visibilité
    U = np.zeros_like(u)
    V = np.zeros_like(v)

    U[masque_u] = np.sign(u[masque_u])
    V[masque_v] = np.sign(v[masque_v])

    # Échantillonnage
    selection = np.zeros_like(fluide)
    selection[::pas, ::pas] = True

    # Création des deux graphiques
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --------------------------------------------------
    # Composante horizontale u
    # --------------------------------------------------

    axes[0].imshow(domaine != 0, cmap="gray_r", origin="upper")

    axes[0].quiver(
        X[masque_u & selection],
        Y[masque_u & selection],
        U[masque_u & selection],
        np.zeros(np.sum(masque_u & selection)),
        color="red",
        angles="xy",
        scale_units="xy",
        scale=0.1,
        width=0.004,
        headwidth=4,
        headlength=5,
        pivot="mid"
    )

    axes[0].set_title("Champ de la composante de vitesse horizontale u")
    axes[0].set_xlabel("Indices colonnes")
    axes[0].set_ylabel("Indices lignes")
    axes[0].set_aspect("equal")

    # --------------------------------------------------
    # Composante verticale v
    # --------------------------------------------------

    axes[1].imshow(domaine != 0, cmap="gray_r", origin="upper")

    axes[1].quiver(
        X[masque_v & selection],
        Y[masque_v & selection],
        np.zeros(np.sum(masque_v & selection)),
        V[masque_v & selection],
        color="blue",
        angles="xy",
        scale_units="xy",
        scale=0.1,
        width=0.004,
        headwidth=4,
        headlength=5,
        pivot="mid"
    )

    axes[1].set_title("Champ de la composante de vitesse verticale v")
    axes[1].set_xlabel("Indices colonnes")
    axes[1].set_ylabel("Indices lignes")
    axes[1].set_aspect("equal")

    plt.savefig("champ_composantes_vitesses.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def afficher_vitesses(u, v, domaine):

    vitesse = np.sqrt(u**2 + v**2)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Affichage du labyrinthe : murs en gris foncé
    murs = (domaine == 0)
    ax.imshow(
        murs,
        cmap=ListedColormap(["white", "dimgray"]),
        origin="upper",
        interpolation="nearest"
    )

    # Masquage des murs pour ne représenter les vitesses
    # que dans les cellules fluides
    vitesse_masquee = np.ma.masked_where(murs, vitesse)

    image = ax.imshow(
        vitesse_masquee,
        cmap="turbo",
        origin="upper",
        interpolation="nearest",
        alpha=0.95
    )

    # Barre de couleurs
    fig.colorbar(image, ax=ax, label="Norme de la vitesse")

    ax.set_title("Distribution spatiale de la vitesse")
    ax.set_xlabel("Indices des colonnes")
    ax.set_ylabel("Indices des lignes")

    plt.savefig("champ_vitesses.png", dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()