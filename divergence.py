import numpy as np

def verification_divergence(u, v, domaine, dx, dy):

    divergence = np.zeros_like(u)

    for i in range(2, u.shape[0]-2):
        for j in range(1, u.shape[1]-1):
            if domaine[i, j] != 0:

                du_dx = (u[i, j+1] - u[i, j]) / dx
                dv_dy = (v[i, j] - v[i-1, j]) / dy
                divergence[i, j] = du_dx + dv_dy

    print("Divergence maximale :", np.max(np.abs(divergence)))

    return divergence