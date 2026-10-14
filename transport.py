import numpy as np

# Calculs flux par reconstruction linéaire 

def calcul_flux(c, u, v, domaine):

    flux_u = np.zeros_like(u, dtype=float)
    flux_v = np.zeros_like(v, dtype=float)


    # Flux à travers les faces verticales
    for i in range(c.shape[0]):
        for j in range(1, c.shape[1]):

            # Ecoulement vers la droite
            if u[i,j] > 0:
                flux_u[i,j] = u[i,j] * c[i,j-1]

            # Ecoulement vers la gauche ou nul
            else:
                flux_u[i,j] = u[i,j] * c[i,j]


    # Flux à travers les faces horizontales
    for i in range(c.shape[0] - 1):
        for j in range(c.shape[1]):

            # Ecoulement vers le bas
            if v[i,j] > 0:
                flux_v[i,j] = v[i,j] * c[i,j]

            # Ecoulement vers le haut ou nul
            else:
                flux_v[i,j] = v[i,j] * c[i+1,j]

    return flux_u, flux_v

# 4. CALCUL DE dc/dt PAR BILAN DE FLUX

def calcul_dc_dt(c, u, v, domaine, dx, dy):

    flux_u, flux_v = calcul_flux(c, u, v, domaine)

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