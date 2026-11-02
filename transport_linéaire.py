import numpy as np
import matplotlib.pyplot as plt
import numba
import matplotlib.colors as colors
import matplotlib.cm as cm

def preparer_masques(domaine):

    interieur = domaine[1:-1, 1:-1]
    fluide = interieur != 0


    sorties = ((interieur == 2) & (domaine[2:, 1:-1] == 0))

    voisins_x = ((domaine[1:-1, :-2] != 0) & (domaine[1:-1, 2:] != 0))
    voisins_y = ((domaine[:-2, 1:-1] != 0) & (domaine[2:, 1:-1] != 0))

    masque_x = fluide & voisins_x
    masque_y = fluide & voisins_y


    faces_x_valides = ((domaine[1:-1, :-2] != 0) & (interieur != 0))
    faces_y_valides = ((domaine[2:, 1:-1] != 0) & (interieur != 0))



    return {
        "masque_x": masque_x,
        "masque_y": masque_y,
        "faces_x_valides": faces_x_valides,
        "faces_y_valides": faces_y_valides,
        "sorties": sorties,
        "fluide" : fluide
    }
"""
@numba.njit(cache=True)
def calcul_gradients(c, dx, dy, masque_x, masque_y):

    dc_dx = np.zeros_like(c)
    dc_dy = np.zeros_like(c)

    return dc_dx, dc_dy
"""

@numba.njit(cache=True)
def calcul_gradients(c, dx, dy, masque_x, masque_y):

    ny, nx = c.shape

    dc_dx = np.zeros((ny, nx), dtype=np.float64)
    dc_dy = np.zeros((ny, nx), dtype=np.float64)

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            # ----------------------------------------
            # Gradient selon x
            # ----------------------------------------

            if masque_x[i - 1, j - 1]:
                dc_dx[i, j] = (c[i, j + 1] - c[i, j - 1]) / (2.0 * dx)

            # ----------------------------------------
            # Gradient selon y
            # ----------------------------------------

            if masque_y[i - 1, j - 1]:
                dc_dy[i, j] = (c[i + 1, j] - c[i - 1, j]) / (2.0 * dy)


    return dc_dx, dc_dy


@numba.njit(cache=True)
def calcul_flux_linéaire(c, u, v, dx, dy,dc_dx, dc_dy,faces_x_valides, faces_y_valides,sorties):

    ny, nx = c.shape

    flux_u = np.zeros_like(u, dtype=np.float64)
    flux_v = np.zeros_like(v, dtype=np.float64)

    # =================================================
    # 1. FLUX À TRAVERS LES FACES VERTICALES
    # =================================================

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            if faces_x_valides[i - 1, j - 1]:
                vitesse = u[i, j]

                if vitesse > 0.0:
                    concentration_face = (c[i, j - 1] + dc_dx[i, j - 1] * dx / 2.0)

                elif vitesse < 0.0:
                    concentration_face = (c[i, j] - dc_dx[i, j] * dx / 2.0)

                else:
                    concentration_face = 0.0

                flux_u[i, j] = vitesse * concentration_face

    # =================================================
    # 2. FLUX À TRAVERS LES FACES HORIZONTALES
    # =================================================

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            if faces_y_valides[i - 1, j - 1]:
                vitesse = v[i, j]

                if vitesse > 0.0:
                    concentration_face = (c[i, j] + dc_dy[i, j] * dy / 2.0)

                elif vitesse < 0.0:
                    concentration_face = (c[i+1, j] - dc_dy[i+1, j] * dy / 2.0)

                else:
                    concentration_face = 0.0

                flux_v[i, j] = vitesse * concentration_face

    return flux_u, flux_v

@numba.njit(cache=True)
def dc_dt_linéaire(c, u, v, dx, dy, masque_x, masque_y, faces_x_valides, faces_y_valides, sorties, fluide):

    ny, nx = c.shape
    dc_dx, dc_dy = calcul_gradients(c, dx, dy, masque_x, masque_y)
    flux_u, flux_v = calcul_flux_linéaire(c, u, v, dx, dy, dc_dx, dc_dy, faces_x_valides, faces_y_valides, sorties)
    dc_dt = np.zeros_like(c, dtype=np.float64)

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            if fluide[i - 1, j - 1]:
                divergence_x = (flux_u[i, j + 1] - flux_u[i, j]) / dx

                divergence_y = (flux_v[i, j] - flux_v[i-1, j]) / dy

                dc_dt[i, j] = -(divergence_x + divergence_y)

    return dc_dt


def runge_kutta(c, u, v, domaine, dx, dy, dt, nombre_pas, instants_affichage):

    instants_affichage = set(instants_affichage)

    masques = preparer_masques(domaine)
    masque_x = masques["masque_x"]
    masque_y = masques["masque_y"]
    fluide = masques ["fluide"]

    faces_x_valides = masques["faces_x_valides"]
    faces_y_valides = masques["faces_y_valides"]

    sorties = masques["sorties"]
 
    entrees = np.argwhere((domaine[1:, :] == 2) & (domaine[:-1, :] == 0))
    entrees[:, 0] += 1

    c[entrees[:, 0], entrees[:, 1]] = 1.0
    resultats_c_kutta = {0: c.copy()}


    for n in range(1, nombre_pas + 1):
        c_1 = c + dt * dc_dt_linéaire(c, u, v, dx, dy, masque_x, masque_y, faces_x_valides, faces_y_valides, sorties, fluide)
        c_1[entrees[:, 0], entrees[:, 1]] = 1.0
        c_2 = c + dt * dc_dt_linéaire(c_1, u, v, dx, dy, masque_x, masque_y, faces_x_valides, faces_y_valides, sorties, fluide)
        c = (c_1 + c_2) / 2

        c[entrees[:, 0], entrees[:, 1]] = 1.0

        if n in instants_affichage:
            resultats_c_kutta[n] = c.copy()

    return resultats_c_kutta

def afficher_concentrations_kutta(resultats_c_kutta, domaine, dt, instants_affichage):

    # =================================================
    # 1. Définition de la palette de couleurs
    # =================================================

    cmap = plt.get_cmap("viridis").copy()

    cmap.set_under("blue")   # c < -0.1
    cmap.set_over("red")     # c > 1.1

    vmin = -0.05
    vmax = 1.05

    # =================================================
    # 2. Création de la figure
    # =================================================

    fig = plt.figure(figsize=(15, 5), layout="constrained")

    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1])
    ]

    cax = fig.add_subplot(gs[0, 2])

    # Titre général commun aux deux graphiques
    fig.suptitle(
        "Résolution par Runge-Kutta et reconstruction linéaire",
        fontsize=16
    )

    # =================================================
    # 3. Affichage des concentrations
    # =================================================

    for ax, n in zip(axes, instants_affichage):

        c_affichage = resultats_c_kutta[n]

        # Masquer les cellules correspondant aux murs
        c_affichage = np.ma.masked_where(domaine == 0, c_affichage)

        image = ax.imshow(
            c_affichage,
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )

        # Titre individuel
        ax.set_title(f"Temps simulé : t = {n * dt:.1f} s")

        # Noms des axes
        ax.set_xlabel("Indices des colonnes")
        ax.set_ylabel("Indices des lignes")

    # =================================================
    # 4. Barre de couleur commune
    # =================================================

    fig.colorbar(
        image,
        cax=cax,
        label="Concentration c",
        extend="both"
    )

    plt.savefig("Résolution par Runge-Kutta et reconstruction linéaire.png", dpi=300, bbox_inches="tight")
    plt.show()