import numba
import numpy as np
import matplotlib.pyplot as plt


@numba.njit(cache=True)
def calcul_flux(c, u, v, domaine):

    flux_u = np.zeros_like(u, dtype=np.float64)
    flux_v = np.zeros_like(v, dtype=np.float64)

    nx, ny = c.shape

    for i in range(1, nx-1):
        for j in range(1, ny-1):

            if domaine[i, j] == 1 or domaine[i, j] == 2:

                # Flux horizontal
                if u[i, j] > 0:
                    c_face = c[i, j-1]
                else:
                    c_face = c[i, j]

                flux_u[i, j] = u[i, j] * c_face

                # Flux vertical
                if domaine[i-1, j] == 0 and domaine[i, j] == 2:
                    c[i, j] = 1.0

                if v[i, j] > 0:
                    c_face = c[i, j]
                else:
                    c_face = c[i+1, j]

                flux_v[i, j] = v[i, j] * c_face

    return flux_u, flux_v


@numba.njit(cache=True)
def dc_dt_cst(c, u, v, domaine, dx, dy):

    flux_u, flux_v = calcul_flux(c, u, v, domaine)

    dc_dt = np.zeros_like(c, dtype=np.float64)

    nx, ny = c.shape

    for i in range(1, nx-1):
        for j in range(1, ny-1):

            if domaine[i, j] == 1 or domaine[i, j] == 2:

                flux_x = (flux_u[i, j+1] - flux_u[i, j]) / dx

                flux_y = (flux_v[i, j] - flux_v[i-1, j]) / dy

                dc_dt[i, j] = -(flux_x + flux_y)

    return dc_dt


def euler_explicite(c, u, v, domaine, dx, dy, dt, nombre_pas, instants_affichage):
    
    resultats_c_euler = {0: c.copy()}
    
    for n in range(1, nombre_pas + 1):

        c = c + dt * dc_dt_cst(c, u, v, domaine, dx, dy)
    
        # On sauvegarde les résultats aux instants voulus
        if n in instants_affichage:
            resultats_c_euler[n] = c.copy()

    return resultats_c_euler


def afficher_concentrations_euler(resultats_c_euler, domaine, dt, instants_affichage):

    # =================================================
    # 1. Définition de la palette de couleurs
    # =================================================

    cmap = plt.get_cmap("viridis").copy()

    cmap.set_under("blue")   # c < -0.05
    cmap.set_over("red")     # c > 1.05

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
        "Résolution par Euler et reconstruction constante",
        fontsize=16
    )

    # =================================================
    # 3. Affichage des concentrations
    # =================================================

    for ax, n in zip(axes, instants_affichage):

        c_affichage = resultats_c_euler[n]

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

    plt.savefig(
        "Résolution par Euler et reconstruction constante.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()