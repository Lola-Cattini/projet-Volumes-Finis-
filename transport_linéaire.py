import numpy as np
import matplotlib.pyplot as plt
import numba

def preparer_masques(domaine):

    # Cellules situées à l'intérieur du domaine
    interieur = domaine[1:-1, 1:-1]

    # Cellules appartenant au chemin
    fluide = interieur != 0

    # ------------------------------------------------
    # Masques pour le calcul des gradients
    # ------------------------------------------------

    voisins_x = ((domaine[1:-1, :-2] != 0) & (domaine[1:-1, 2:] != 0))

    voisins_y = ((domaine[:-2, 1:-1] != 0) & (domaine[2:, 1:-1] != 0))

    sorties = ((interieur == 2) & (domaine[2:, 1:-1] == 0))

    masque_x = fluide & voisins_x

    masque_y = fluide & (voisins_y | sorties)

    # ------------------------------------------------
    # Masques pour le calcul des flux
    # ------------------------------------------------

    faces_x_valides = ((domaine[1:-1, :-2] != 0) & (interieur != 0))

    faces_y_valides = ((domaine[:-2, 1:-1] != 0) & (interieur != 0))

    # ------------------------------------------------
    # Masque des cellules où l'équation de transport doit être calculée
    # ------------------------------------------------

    masque_transport = ((interieur == 1) | (interieur == 2))

    return {
        "masque_x": masque_x,
        "masque_y": masque_y,
        "faces_x_valides": faces_x_valides,
        "faces_y_valides": faces_y_valides,
        "sorties": sorties,
        "masque_transport": masque_transport
    }

@numba.njit(cache=True)
def limiteur_MC_scalaire(gauche, centre, droite):
    """Limiteur MC appliqué à une seule valeur."""

    if gauche * droite > 0.0:
        valeur = min(abs(centre), 2.0 * abs(gauche), 2.0 * abs(droite))

        if centre > 0.0:
            return valeur
        elif centre < 0.0:
            return -valeur

    return 0.0


@numba.njit(cache=True)
def calcul_gradients_numba(c, dx, dy, masque_x, masque_y):

    ny, nx = c.shape

    dc_dx = np.zeros((ny, nx), dtype=np.float64)
    dc_dy = np.zeros((ny, nx), dtype=np.float64)

    # Parcours des cellules intérieures
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            # ----------------------------------------
            # Gradient selon x
            # ----------------------------------------

            if masque_x[i - 1, j - 1]:

                gauche = (c[i, j] - c[i, j - 1]) / dx

                centre = (c[i, j + 1] - c[i, j - 1]) / (2.0 * dx)

                droite = (c[i, j + 1] - c[i, j]) / dx

                dc_dx[i, j] = limiteur_MC_scalaire(gauche, centre, droite)

            # ----------------------------------------
            # Gradient selon y
            # ----------------------------------------

            if masque_y[i - 1, j - 1]:

                haut = (c[i, j] - c[i - 1, j]) / dy

                centre = (c[i + 1, j] - c[i - 1, j]) / (2.0 * dy)

                bas = (c[i + 1, j] - c[i, j]) / dy

                dc_dy[i, j] = limiteur_MC_scalaire(haut, centre, bas)

    return dc_dx, dc_dy


@numba.njit(cache=True)
def dc_dt_numba(c, u, v, dx, dy, masque_x, masque_y, faces_x_valides, faces_y_valides, sorties, masque_transport):

    ny, nx = c.shape

    dc_dx, dc_dy = calcul_gradients_numba(c, dx, dy, masque_x, masque_y)

    flux_u = np.zeros_like(u, dtype=np.float64)
    flux_v = np.zeros_like(v, dtype=np.float64)

    dc_dt = np.zeros_like(c, dtype=np.float64)

    # =================================================
    # 1. FLUX À TRAVERS LES FACES VERTICALES
    # =================================================

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            if faces_x_valides[i - 1, j - 1]:

                vitesse = u[i, j]

                if vitesse > 0.0:
                    concentration_face = (c[i, j - 1]+ dc_dx[i, j - 1] * dx / 2.0)

                elif vitesse < 0.0:
                    concentration_face = (c[i, j]- dc_dx[i, j] * dx / 2.0)

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
                    concentration_face = (c[i - 1, j]+ dc_dy[i - 1, j] * dy / 2.0)

                elif vitesse < 0.0:
                    concentration_face = (c[i, j]- dc_dy[i, j] * dy / 2.0)

                else:
                    concentration_face = 0.0

                flux_v[i, j] = vitesse * concentration_face

    # =================================================
    # 3. FLUX SORTANT PAR LE BAS
    # =================================================

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            if sorties[i - 1, j - 1]:

                vitesse = v[i, j]

                if vitesse > 0.0:
                    flux_v[i + 1, j] = vitesse * c[i, j]
                else:
                    flux_v[i + 1, j] = 0.0

    # =================================================
    # 4. DIVERGENCE ET ÉQUATION DE TRANSPORT
    # =================================================

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            if masque_transport[i - 1, j - 1]:

                divergence_x = (flux_u[i, j + 1] - flux_u[i, j]) / dx

                divergence_y = (flux_v[i + 1, j] - flux_v[i, j]) / dy

                dc_dt[i, j] = -(divergence_x + divergence_y)

    return dc_dt


def runge_kutta(c, u, v, domaine, dx, dy, dt, nombre_pas, instants_affichage):

    resultats_c_kutta = {0: c.copy()}
    instants_affichage = set(instants_affichage)

    # Pré-calcul des masques une seule fois
    masques = preparer_masques(domaine)
    masque_x = masques["masque_x"]
    masque_y = masques["masque_y"]

    faces_x_valides = masques["faces_x_valides"]
    faces_y_valides = masques["faces_y_valides"]

    sorties = masques["sorties"]
    masque_transport = masques["masque_transport"]

    entrees = np.argwhere((domaine[1:, :] == 2) & (domaine[:-1, :] == 0))
    entrees[:, 0] += 1

    c[entrees[:, 0], entrees[:, 1]] = 1.0


    for n in range(1, nombre_pas + 1):

        c_1 = c + dt * dc_dt_numba(c, u, v, dx, dy, masque_x, masque_y, faces_x_valides, faces_y_valides, sorties, masque_transport)

        c_2 = c + dt * dc_dt_numba(c_1, u, v, dx, dy, masque_x, masque_y, faces_x_valides, faces_y_valides, sorties, masque_transport)

        c = (c_1 + c_2) / 2

        # Condition à l'entrée au début du pas de temps
        c[entrees[:, 0], entrees[:, 1]] = 1.0
        if n in instants_affichage:
            resultats_c_kutta[n] = c.copy()

    return resultats_c_kutta


def afficher_concentrations_kutta(resultats_c_kutta, domaine, dt, instants_affichage):

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, n in zip(axes, instants_affichage):
        c_affichage = resultats_c_kutta[n]
        c_affichage = np.ma.masked_where(domaine == 0, c_affichage)

        image = ax.imshow(c_affichage, origin='upper', cmap='viridis', vmin=0, vmax=1)

        ax.set_title(f"t = {n * dt:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(image, ax=axes, label="Concentration c")

    plt.show()