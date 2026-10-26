import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. CALCUL DES GRADIENTS DE CONCENTRATION
# ============================================================
"""
def calcul_gradients_lineaires(c, domaine, dx, dy):

    dc_dx = np.zeros_like(c, dtype=float)
    dc_dy = np.zeros_like(c, dtype=float)

    ny, nx = c.shape

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):

            # On calcule les gradients uniquement dans le fluide
            if domaine[i, j] == 0:
                continue

            # ------------------------------------------------
            # Gradient selon x
            # ------------------------------------------------
            
            # CAS 2 : chemin horizontal
            if domaine[i, j-1] == 0 and domaine[i, j] == 2:
                c[i,j]= 1.0
                        
            if domaine [i,j-1] != 0 and domaine [i,j+1] != 0:
                dc_dx[i, j] = (c[i, j+1] - c[i, j-1]) / (2 * dx)


            # ------------------------------------------------
            # Gradient selon y
            # ------------------------------------------------



            if domaine [i-1,j] != 0 and domaine [i+1,j] != 0:
                dc_dy[i, j] = (c[i+1, j] - c[i-1, j]) / (2 * dy)

    return dc_dx, dc_dy
"""

def limiteur_MC(gauche, centre, droite):
    """
    Limiteur de pente MC.
    Réduit les oscillations près des variations brusques
    tout en conservant une reconstruction linéaire.
    """

    return np.where(
        gauche * droite > 0,
        np.sign(centre) * np.minimum(
            np.abs(centre),
            np.minimum(
                2 * np.abs(gauche),
                2 * np.abs(droite)
            )
        ),
        0.0
    )


def calcul_gradients_lineaires(c, domaine, dx, dy):

    dc_dx = np.zeros_like(c, dtype=float)
    dc_dy = np.zeros_like(c, dtype=float)

    fluide = domaine[1:-1, 1:-1] != 0

    # ------------------------------------------------
    # Gradient selon x
    # ------------------------------------------------

    # Cellules fluides dont les deux voisins horizontaux sont fluides
    voisins_x = (
        (domaine[1:-1, :-2] != 0)
        & (domaine[1:-1, 2:] != 0)
    )

    masque_x = fluide & voisins_x

    # Gradients entre les mailles voisines
    gradient_gauche_x = (
        c[1:-1, 1:-1] - c[1:-1, :-2]
    ) / dx

    gradient_droite_x = (
        c[1:-1, 2:] - c[1:-1, 1:-1]
    ) / dx

    # Gradient centré, comme dans ta méthode initiale
    gradient_centre_x = (
        c[1:-1, 2:] - c[1:-1, :-2]
    ) / (2 * dx)

    # Limitation de la pente
    gradient_x = limiteur_MC(
        gradient_gauche_x,
        gradient_centre_x,
        gradient_droite_x
    )

    dc_dx[1:-1, 1:-1] = np.where(
        masque_x,
        gradient_x,
        0.0
    )

    # ------------------------------------------------
    # Gradient selon y
    # ------------------------------------------------

    # Cellules fluides dont les deux voisins verticaux sont fluides
    voisins_y = (
        (domaine[:-2, 1:-1] != 0)
        & (domaine[2:, 1:-1] != 0)
    )

    sorties = (
        (domaine[1:-1, 1:-1] == 2)
        & (domaine[2:, 1:-1] == 0)
    )

    masque_y = fluide & (voisins_y | sorties)

    # Gradients entre les mailles voisines
    gradient_haut_y = (
        c[1:-1, 1:-1] - c[:-2, 1:-1]
    ) / dy

    gradient_bas_y = (
        c[2:, 1:-1] - c[1:-1, 1:-1]
    ) / dy

    # Gradient centré, comme dans ta méthode initiale
    gradient_centre_y = (
        c[2:, 1:-1] - c[:-2, 1:-1]
    ) / (2 * dy)

    # Limitation de la pente
    gradient_y = limiteur_MC(
        gradient_haut_y,
        gradient_centre_y,
        gradient_bas_y
    )

    dc_dy[1:-1, 1:-1] = np.where(
        masque_y,
        gradient_y,
        0.0
    )

    return dc_dx, dc_dy

# ============================================================
# 2. CALCUL DES FLUX PAR RECONSTRUCTION LINEAIRE
# ============================================================
"""
def calcul_flux_lineaire(c, u, v, domaine, dx, dy):

    flux_u = np.zeros_like(u, dtype=float)
    flux_v = np.zeros_like(v, dtype=float)

    dc_dx, dc_dy = calcul_gradients_lineaires(c, domaine, dx, dy)

    ny, nx = c.shape

    # --------------------------------------------------------
    # Flux à travers les faces verticales
    # --------------------------------------------------------

    for i in range(1,ny-1):
        for j in range(1, nx-1):
            
            # Cas 2 : chemin horizontal
            if domaine [i,j-1] == 0 and domaine [i,j] == 2:
                c_face = 1
                flux_u[i, j] = u[i, j] * c_face
                continue
            
            # Écoulement vers la droite
            if u[i, j] > 0:

                c_face = c[i, j-1] + dc_dx[i, j-1] * dx / 2
                flux_u[i, j] = u[i, j] * c_face

            # Écoulement vers la gauche
            elif u[i, j] < 0:

                c_face = c[i, j] - dc_dx[i, j] * dx / 2
                flux_u[i, j] = u[i, j] * c_face

            else:
                flux_u[i, j] = 0


    # --------------------------------------------------------
    # Flux à travers les faces horizontales
    # --------------------------------------------------------

    for i in range(1,ny-1):
        for j in range(1,nx-1):

            # Cas 1 : chemin vertical + général
            if domaine [i-1,j] == 0 and domaine [i,j] == 2:
                c_face = 1
                flux_v[i, j] = v[i, j] * c_face
                continue

            # Écoulement vers le bas
            if v[i, j] > 0:
            
                c_face = c[i-1, j]+ dc_dy[i-1, j] * dy / 2
                flux_v[i, j] = v[i, j] * c_face

            # Écoulement vers le haut
            elif v[i, j] < 0:
            
                c_face = c[i, j]- dc_dy[i, j] * dy / 2
                flux_v[i, j] = v[i, j] * c_face

            else:
                flux_v[i, j] = 0

    return flux_u, flux_v
"""
def calcul_flux_lineaire(c, u, v, domaine, dx, dy):

    flux_u = np.zeros_like(u, dtype=float)
    flux_v = np.zeros_like(v, dtype=float)

    dc_dx, dc_dy = calcul_gradients_lineaires(
        c, domaine, dx, dy
    )

    zone = np.s_[1:-1, 1:-1]

    # ========================================================
    # 1. FLUX À TRAVERS LES FACES VERTICALES
    # ========================================================

    u_int = u[zone]

    # Cellules situées à gauche et à droite de chaque face
    domaine_gauche = domaine[1:-1, :-2]
    domaine_droite = domaine[1:-1, 1:-1]

    # Une face est autorisée uniquement si elle sépare
    # deux cellules appartenant au chemin.
    faces_x_valides = (
        (domaine_gauche != 0)
        & (domaine_droite != 0)
    )

    # Reconstruction linéaire à gauche de la face
    c_face_droite = (
        c[1:-1, :-2]
        + dc_dx[1:-1, :-2] * dx / 2
    )

    # Reconstruction linéaire à droite de la face
    c_face_gauche = (
        c[1:-1, 1:-1]
        - dc_dx[1:-1, 1:-1] * dx / 2
    )

    flux_x = np.where(
        u_int > 0,
        u_int * c_face_droite,
        np.where(
            u_int < 0,
            u_int * c_face_gauche,
            0.0
        )
    )

    # Annuler les flux traversant les murs
    flux_u[zone] = np.where(
        faces_x_valides,
        flux_x,
        0.0
    )

    # ========================================================
    # 2. FLUX À TRAVERS LES FACES HORIZONTALES
    # ========================================================

    v_int = v[zone]

    # Cellules situées au-dessus et en dessous de chaque face
    domaine_haut = domaine[:-2, 1:-1]
    domaine_bas = domaine[1:-1, 1:-1]

    faces_y_valides = (
        (domaine_haut != 0)
        & (domaine_bas != 0)
    )

    # Reconstruction linéaire au-dessus de la face
    c_face_bas = (
        c[:-2, 1:-1]
        + dc_dy[:-2, 1:-1] * dy / 2
    )

    # Reconstruction linéaire en dessous de la face
    c_face_haut = (
        c[1:-1, 1:-1]
        - dc_dy[1:-1, 1:-1] * dy / 2
    )

    flux_y = np.where(
        v_int > 0,
        v_int * c_face_bas,
        np.where(
            v_int < 0,
            v_int * c_face_haut,
            0.0
        )
    )

    # Flux uniquement entre deux cellules du chemin
    flux_v[zone] = np.where(faces_y_valides, flux_y, 0.0)

    # --------------------------------------------------------
    # Prise en compte du flux sortant par le bas du domaine
    # --------------------------------------------------------

    sorties = (
        (domaine[1:-1, 1:-1] == 2)
        & (domaine[2:, 1:-1] == 0)
    )

    indices_sorties = np.argwhere(sorties)

    for i_local, j_local in indices_sorties:
        i = i_local + 1
        j = j_local + 1

        # Flux sortant uniquement si la vitesse est dirigée vers le bas
        flux_v[i + 1, j] = max(v[i, j], 0.0) * c[i, j]

    return flux_u, flux_v
    
"""
def dc_dt_lineaire(c, u, v, domaine, dx, dy):

    flux_u, flux_v = calcul_flux_lineaire(c, u, v, domaine, dx, dy)

    dc_dt = np.zeros_like(c, dtype=float)

    for i in range(1, c.shape[0] - 1):
        for j in range(1, c.shape[1] - 1):

            if domaine[i,j] == 1 or domaine[i,j] == 2:

                # Flux horizontal :
                flux_x = (flux_u[i,j+1] - flux_u[i,j])

                # Flux vertical :
                flux_y = (flux_v[i+1,j] - flux_v[i,j])

                # Conservation de la concentration
                dc_dt[i,j] = -(flux_x / dx + flux_y / dy)

    return dc_dt
"""

def dc_dt_lineaire(c, u, v, domaine, dx, dy):

    flux_u, flux_v = calcul_flux_lineaire(
        c, u, v, domaine, dx, dy
    )

    dc_dt = np.zeros_like(c, dtype=float)

    # Masque des cellules appartenant au chemin
    masque = (
        (domaine[1:-1, 1:-1] == 1)
        | (domaine[1:-1, 1:-1] == 2)
    )

    # Divergence horizontale
    flux_x = (
        flux_u[1:-1, 2:]
        - flux_u[1:-1, 1:-1]
    ) / dx

    # Divergence verticale
    flux_y = (
        flux_v[2:, 1:-1]
        - flux_v[1:-1, 1:-1]
    ) / dy

    # Équation de transport
    dc_dt[1:-1, 1:-1] = np.where(
        masque,
        -(flux_x + flux_y),
        0.0
    )

    return dc_dt


def runge_kutta(c, u, v, domaine, dx, dy, dt, nombre_pas, instants_affichage):

    resultats_c_kutta = {0: c.copy()}
    instants_affichage = set(instants_affichage)

    entrees = np.argwhere(
    (domaine[1:, :] == 2) & (domaine[:-1, :] == 0))
    entrees[:, 0] += 1

    c[entrees[:, 0], entrees[:, 1]] = 1.0

    for n in range(1, nombre_pas + 1):

        c_1 = c + dt * dc_dt_lineaire(c, u, v, domaine, dx, dy)

        c_2 = c + dt * dc_dt_lineaire(c_1, u, v, domaine, dx, dy)

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
    