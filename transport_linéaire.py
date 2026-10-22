import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CALCUL DES GRADIENTS DE CONCENTRATION
# ============================================================

def calcul_gradients_lineaires(c, domaine, dx, dy):

    dc_dx = np.zeros_like(c, dtype=float)
    dc_dy = np.zeros_like(c, dtype=float)

    ny, nx = c.shape

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            # On calcule les gradients uniquement dans le fluide
            if domaine[i, j] == 0:
                dc_dx[i, j] = 0
                dc_dy[i, j] = 0

            # ------------------------------------------------
            # Gradient selon x
            # ------------------------------------------------

            if domaine [i,j-1] and domaine [i,j+1] != 0:
                dc_dx[i, j] = (c[i, j+1] - c[i, j-1]) / (2 * dx)

            if domaine [i,j-1] == 0:
                dc_dx[i,j] = 0

            if domaine[i,j+1] == 0:
                dc_dx[i,j] = 0

            # ------------------------------------------------
            # Gradient selon y
            # ------------------------------------------------

            if domaine [i-1,j] and domaine [i+1,j] != 0:
                dc_dy[i, j] = (c[i+1, j] - c[i-1, j]) / (2 * dy)

            if domaine [i-1,j] == 0:
                dc_dy[i,j] = 0

            if domaine [i+1,j] == 0:
                dc_dy[i,j] = 0

    return dc_dx, dc_dy


# ============================================================
# 2. CALCUL DES FLUX PAR RECONSTRUCTION LINEAIRE
# ============================================================

def calcul_flux_lineaire(c, u, v, domaine, dx, dy):

    flux_u = np.zeros_like(u, dtype=float)
    flux_v = np.zeros_like(v, dtype=float)

    dc_dx, dc_dy = calcul_gradients_lineaires(c, domaine, dx, dy)

    ny, nx = c.shape

    # --------------------------------------------------------
    # Flux à travers les faces verticales
    # --------------------------------------------------------

    for i in range(ny):
        for j in range(1, nx):

            # Écoulement vers la droite
            if u[i, j] > 0:
                c_face = (c[i, j-1]+ dc_dx[i, j-1] * dx / 2)
                flux_u[i, j] = u[i, j] * c_face

            # Écoulement vers la gauche
            elif u[i, j] < 0:
                c_face = (c[i, j]- dc_dx[i, j] * dx / 2)
                flux_u[i, j] = u[i, j] * c_face

            else:
                flux_u[i, j] = 0


    # --------------------------------------------------------
    # Flux à travers les faces horizontales
    # --------------------------------------------------------

    for i in range(ny-1):
        for j in range(1,nx):

            # Écoulement vers la droite
            if v[i, j] > 0:
                c_face = (c[i, j-1]+ dc_dy[i, j-1] * dy / 2)
                flux_v[i, j] = v[i, j] * c_face

            # Écoulement vers la gauche
            elif v[i, j] < 0:
                c_face = (c[i, j]- dc_dy[i, j] * dy / 2)
                flux_v[i, j] = v[i, j] * c_face

            else:
                flux_v[i, j] = 0

    return flux_u, flux_v

def dc_dt_lineaire(c, u, v, domaine, dx, dy):

    flux_u, flux_v = calcul_flux_lineaire(c, u, v, domaine, dx, dy)

    dc_dt = np.zeros_like(c, dtype=float)

    for i in range(1, c.shape[0] - 1):
        for j in range(1, c.shape[1] - 1):

            if domaine[i,j] == 1 or domaine[i,j] == 2:

                # Flux horizontal :
                flux_x = (flux_u[i,j+1] - flux_u[i,j])

                # Flux vertical :
                flux_y = (flux_v[i,j] - flux_v[i-1,j])

                # Conservation de la concentration
                dc_dt[i,j] = -(flux_x / dx + flux_y / dy)

    return dc_dt


def runge_kutta (c, u, v, domaine, dx, dy, dt, nombre_pas, instants_affichage):
    resultats_c_kutta = {0: c.copy()}

    for n in range(1, nombre_pas + 1):

        c_1 = c + dt * dc_dt_lineaire(c, u, v, domaine, dx, dy)
        c_2 = c + dt * dc_dt_lineaire(c_1, u, v, domaine, dx, dy)
        c_final = (c_1 + c_2)/2

        if n in instants_affichage:
            resultats_c_kutta[n] = c_final.copy()

    return resultats_c_kutta


def afficher_concentrations_kutta(resultats_c_kutta, domaine, dt, instants_affichage):

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, n in zip(axes, instants_affichage):
        c_affichage = resultats_c_kutta[n]
        c_affichage = np.ma.masked_where(domaine == 0, c_affichage)

        image = ax.imshow(
            c_affichage,
            origin='upper',
            cmap='viridis',
            vmin=0,
            vmax=1
        )

        ax.set_title(f"t = {n * dt:.1f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.colorbar(image, ax=axes, label="Concentration c")

    plt.show()