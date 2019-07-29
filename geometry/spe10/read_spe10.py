import numpy as np
import matplotlib.pyplot as plt

def main(layer_num, plot=False):
    nx, ny, nz = 60, 220, 85
    n = nx*ny
    N = n*nz

    perm = np.loadtxt("spe_perm.dat").ravel()
    k = np.array([perm[:N], perm[N:2*N], perm[2*N:]]).T

    # loop layer by layer
    for layer in np.atleast_1d(layer_num):
        # extract the permeability
        k_layer = k[layer*n:(layer+1)*n, :]

        # save the permeability on file
        file_name = "perm/" + str(layer) + ".tar.gz"
        np.savetxt(file_name, k_layer, delimiter=",")

        if plot:
            k_layer = np.log10(k_layer[:, 0].reshape((ny, nx)).T)
            plt.imshow(k_layer)
            plt.show()


if __name__ == "__main__":

    layer_id = np.arange(85)
    main(layer_id, plot=False)
