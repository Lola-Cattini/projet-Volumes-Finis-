import numpy as np


# ============================================================
# 1. CALCUL DES GRADIENTS DE CONCENTRATION
# ============================================================

def calcul_gradients_lineaires(c, domaine, dx, dy):

    dc_dx = np.zeros_like(c, dtype=float)
    dc_dy = np.zeros_like(c, dtype=float)

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):

            # On calcule les gradients uniquement dans le fluide
            if domaine[i, j] == 0:
                dc_dx[i, j] = 0
                dc_dy[i, j] = 0

            # ------------------------------------------------
            # Gradient selon x
            # ------------------------------------------------

            if domaine [i,j-1] and domaine [i,j+1] != 0:
                dc_dx[i, j] = 

            if domaine [i,j-1] = 0:
                dc_dx[i,j] = 

            if domaine [i,j+1] = 0:
                dc_dx[i,j] =

            # ------------------------------------------------
            # Gradient selon y
            # ------------------------------------------------

            if domaine [i-1,j] and domaine [i+1,j] != 0:
                dc_dy[i, j] = 

            if domaine [i-1,j] = 0:
                dc_dy[i,j] = 

            if domaine [i+1,j] = 0:
                dc_dy[i,j] =

    return dc_dx, dc_dy


# ============================================================
# 2. CALCUL DES FLUX PAR RECONSTRUCTION LINEAIRE
# ============================================================

def calcul_flux_lineaire(c, u, v, domaine, dx, dy):

    flux_u = np.zeros_like(u, dtype=float)
    flux_v = np.zeros_like(v, dtype=float)

    dc_dx, dc_dy = calcul_gradients_lineaires(
        c, domaine, dx, dy
    )

    ny, nx = c.shape

    # --------------------------------------------------------
    # Flux à travers les faces verticales
    # u[i,j] : face entre les cellules (i,j-1) et (i,j)
    # --------------------------------------------------------

    for i in range(ny):
        for j in range(1, nx):

            # Une face ne transporte pas de matière
            # si l'une de ses cellules est solide
            if (
                domaine[i, j-1] not in (1, 2)
                or domaine[i, j] not in (1, 2)
            ):
                continue

            # Écoulement vers la droite
            if u[i, j] > 0:

                # Cellule amont : (i,j-1)
                c_face = (
                    c[i, j-1]
                    + dc_dx[i, j-1] * dx / 2
                )

            # Écoulement vers la gauche
            elif u[i, j] < 0:

                # Cellule amont : (i,j)
                c_face = (
                    c[i, j]
                    - dc_dx[i, j] * dx / 2
                )

            else:
                c_face = 0.0

            flux_u[i, j] = u[i, j] * c_face

    # --------------------------------------------------------
    # Flux à travers les faces horizontales
    # v[i,j] : face entre les cellules (i,j) et (i+1,j)
    # --------------------------------------------------------

    for i in range(ny-1):
        for j in range(nx):

            if (
                domaine[i, j] not in (1, 2)
                or domaine[i+1, j] not in (1, 2)
            ):
                continue

            # Écoulement vers le bas
            if v[i, j] > 0:

                # Cellule amont : (i,j)
                c_face = (
                    c[i, j]
                    + dc_dy[i, j] * dy / 2
                )

            # Écoulement vers le haut
            elif v[i, j] < 0:

                # Cellule amont : (i+1,j)
                c_face = (
                    c[i+1, j]
                    - dc_dy[i+1, j] * dy / 2
                )

            else:
                c_face = 0.0

            flux_v[i, j] = v[i, j] * c_face

    return flux_u, flux_v