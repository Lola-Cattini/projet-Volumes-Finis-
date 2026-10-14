import numpy as np
import matplotlib.pyplot as plt

# Calculs flux par reconstruction constante 

def calcul_flux(c, u, v, domaine):

    flux_u = np.zeros_like(u, dtype=float)
    flux_v = np.zeros_like(v, dtype=float)


    # Flux à travers les faces verticales
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):

            if domaine[i,j] == 1 or domaine[i,j] == 2:

                # Ecoulement vers la droite
                if u[i,j] > 0:
                    flux_u[i,j] = u[i,j] * c[i,j-1]

                # Ecoulement vers la gauche ou nul
                else:
                    flux_u[i,j] = u[i,j] * c[i,j]
            else:
                flux_u[i,j] = 0

    # Flux à travers les faces horizontales
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):

            if domaine[i,j] == 1 or domaine[i,j] == 2:

                if domaine[i-1,j] == 0 and domaine[i,j] == 2:
                    c[i,j] = 1  #condition limite faible

                # Ecoulement vers le bas
                if v[i,j] > 0:
                    flux_v[i,j] = v[i,j] * c[i,j]

                # Ecoulement vers le haut ou nul
                else:
                    flux_v[i,j] = v[i,j] * c[i+1,j]

    return flux_u, flux_v



def dc_dt_cst(c, u, v, domaine, dx, dy):

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


def euler_explicite(c, u, v, domaine, dx, dy, dt, nombre_pas, instants_affichage):
    resultats_c = {0: c.copy()}

    for n in range(1, nombre_pas + 1):

        # Euler explicite
        c = c + dt * dc_dt_cst(c, u, v, domaine, dx, dy)

        # On sauvegarde les résultats aux instants voulus
        if n in instants_affichage:
            resultats_c[n] = c.copy()

    return resultats_c


def afficher_concentrations(resultats_c, domaine, dt, instants_affichage):

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    for ax, n in zip(axes, instants_affichage):
        c_affichage = resultats_c[n]
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


def runge_kutta (c, u, v, domaine, dx, dy, dt, nombre_pas, instants_affichage):
    resultats_c = {0: c.copy()}

    for n in range(1, nombre_pas + 1):

        c_1 = c + dt * dc_dt_cst(c, u, v, domaine, dx, dy)
        c_2 = c + dt * dc_dt_cst(c_1, u, v, domaine, dx, dy)
        c_final = (c_1 + c_2)/2

        if n in instants_affichage:
                    resultats_c[n] = c_final.copy()

    return resultats_c
