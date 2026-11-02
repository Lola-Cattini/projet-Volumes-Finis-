import numpy as np
import matplotlib.pyplot as plt

# Calculs flux par reconstruction constante 

def calcul_flux(c, u, v, domaine):

    flux_u = np.zeros_like(u, dtype=float)
    flux_v = np.zeros_like(v, dtype=float)

    zone = np.s_[1:-1, 1:-1]

    interieur = (domaine[zone] == 1) | (domaine[zone] == 2)

    # =====================================================
    # 1. Flux horizontaux
    # =====================================================

    u_int = u[zone]

    c_gauche = c[1:-1, :-2]
    c_cellule = c[zone]

    concentration_face = np.where(u_int > 0, c_gauche, c_cellule)

    flux_horizontal = u_int * concentration_face

    flux_u[zone] = np.where(interieur, flux_horizontal, 0.0)

    # =====================================================
    # 2. Flux verticaux
    # =====================================================

    # Condition à l'entrée verticale, comme dans ton code actuel
    entree_verticale = ((domaine[:-2, 1:-1] == 0) & (domaine[1:-1, 1:-1] == 2))

    c_int = c[zone]
    c_int[entree_verticale] = 1.0

    v_int = v[zone]

    c_cellule = c[zone]
    c_bas = c[2:, 1:-1]

    concentration_face = np.where(v_int > 0, c_cellule, c_bas)

    flux_vertical = v_int * concentration_face

    # Flux nul en dehors du domaine de calcul
    flux_v[zone] = np.where(interieur, flux_vertical, 0.0)

    return flux_u, flux_v


def dc_dt_cst(c, u, v, domaine, dx, dy):

    flux_u, flux_v = calcul_flux(c, u, v, domaine)

    dc_dt = np.zeros_like(c, dtype=float)
    zone = np.s_[1:-1, 1:-1]

    # Divergence des flux horizontaux
    flux_x = (flux_u[1:-1, 2:] - flux_u[1:-1, 1:-1]) / dx

    # Divergence des flux verticaux
    flux_y = (flux_v[1:-1, 1:-1] - flux_v[:-2, 1:-1]) / dy

    interieur = ((domaine[zone] == 1) | (domaine[zone] == 2))

    dc_dt[zone] = np.where(interieur, -(flux_x + flux_y), 0.0)

    return dc_dt

def euler_explicite(c, u, v, domaine, dx, dy, dt, nombre_pas, instants_affichage):
    resultats_c_euler = {0: c.copy()}
    

    for n in range(1, nombre_pas + 1):

        # Euler explicite
        c = c + dt * dc_dt_cst(c, u, v, domaine, dx, dy)
    
        # On sauvegarde les résultats aux instants voulus
        if n in instants_affichage:
            resultats_c_euler[n] = c.copy()

    return resultats_c_euler


def afficher_concentrations_euler(resultats_c_euler, domaine, dt, instants_affichage):

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    fig.suptitle("Résolution par Euler et reconstruction constante", fontsize=16, fontweight='bold')

    for ax, n in zip(axes, instants_affichage):
        c_affichage = resultats_c_euler[n]
        c_affichage = np.ma.masked_where(domaine == 0, c_affichage)

        image = ax.imshow( c_affichage, origin='upper', cmap='viridis', vmin=0, vmax=1)

        ax.set_title(f"Temps : t = {n * dt:.1f} s")
        ax.set_xlabel("Indices des colonnes")
        ax.set_ylabel("Indices des lignes")

    # Réserver un espace à droite pour la barre de couleurs
    fig.subplots_adjust(right=0.88, top=0.85, wspace=0.35)

    # Ajouter la barre de couleurs à droite
    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    fig.colorbar(image, cax=cbar_ax, label="Concentration c")

    plt.savefig("Résolution par Euler et reconstruction constante.png", dpi=300, bbox_inches="tight")

    plt.show()