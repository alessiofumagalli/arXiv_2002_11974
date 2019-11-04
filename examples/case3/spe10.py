import numpy as np
from scipy.stats import hmean
import porepy as pp

# ------------------------------------------------------------------------------#

class Spe10(object):

# ------------------------------------------------------------------------------#

    def __init__(self, layers):
        self.full_shape = (60, 220, 85)
        #self.full_shape = (60, 50, 85)
        self.full_physdims = (365.76, 670.56, 51.816)

        self.layers = np.sort(np.atleast_1d(layers))

        self._compute_size()
        self._create_gb()

        self.perm = None
        self.layers_id = None
        self.partition = np.arange(self.N)

# ------------------------------------------------------------------------------#

    def _compute_size(self):
        dim = self.layers.size
        if dim == 1:
            self.shape = list(self.full_shape[:2])
            self.physdims = list(self.full_physdims[:2])
        else:
            self.shape = list(self.full_shape[:2]) + [dim]
            thickness = self.full_physdims[2] / self.full_shape[2] * dim
            self.physdims = list(self.full_physdims[:2]) + [thickness]

        self.N = np.prod(self.shape)
        self.n = np.prod(self.shape[:2])

# ------------------------------------------------------------------------------#

    def _create_gb(self,):
        g = pp.CartGrid(self.shape, self.physdims)
        g.compute_geometry()

        # it's only one grid but the solver is build on a gb
        self.gb = pp.meshing.grid_list_to_grid_bucket([g])
        self.gb.add_porepy_props()

# ------------------------------------------------------------------------------#

    def read_perm(self, perm_folder):

        shape = (self.n, self.layers.size)
        perm_xx, perm_yy, perm_zz = np.empty(shape), np.empty(shape), np.empty(shape)
        layers_id = np.empty(shape)

        for pos, layer in enumerate(self.layers):
            perm_file = perm_folder + str(layer) + ".tar.gz"
            #perm_file = perm_folder + "small_0.csv"
            perm_layer = np.loadtxt(perm_file, delimiter=",")
            perm_xx[:, pos] = perm_layer[:, 0]
            perm_yy[:, pos] = perm_layer[:, 1]
            perm_zz[:, pos] = perm_layer[:, 2]
            layers_id[:, pos] = layer

        shape = self.n*self.layers.size
        perm_xx = perm_xx.reshape(shape, order="F")
        perm_yy = perm_yy.reshape(shape, order="F")
        perm_zz = perm_zz.reshape(shape, order="F")
        self.perm = np.stack((perm_xx, perm_yy, perm_zz)).T

        self.layers_id = layers_id.reshape(shape, order="F")

# ------------------------------------------------------------------------------#

    def coarsen(self, cdepth=3):
        if self.perm is None:
            raise ValueError("Permeability should not be set")

        dim = self.gb.dim_max()
        if dim == 2:
            perm = pp.SecondOrderTensor(kxx=self.perm[:, 0],
                                        kyy=self.perm[:, 1],
                                        kzz=1)
        else:
            perm = pp.SecondOrderTensor(kxx=self.perm[:, 0],
                                        kyy=self.perm[:, 1],
                                        kzz=self.perm[:, 2])

        matrix = pp.coarsening.tpfa_matrix(self.gb, perm)
        partition = pp.coarsening.create_partition(matrix, self.gb, cdepth=cdepth)
        pp.coarsening.generate_coarse_grid(self.gb, partition)

        # we need to map the permeability and layer id according to a possibile coarsening
        for g, _ in self.gb:
            cell_map = partition[g][1]
            perm = np.zeros((np.amax(cell_map) + 1, 3))
            layers_id = np.zeros(perm.shape[0])

            for idx in np.arange(perm.shape[0]):
                perm[idx] = hmean(self.perm[cell_map == idx])
                layers_id[idx] = np.mean(self.layers_id[cell_map == idx])

        self.perm = perm
        self.layers_id = layers_id

# ------------------------------------------------------------------------------#

    def save_perm(self):

        names = ["log10_perm_xx", "log10_perm_yy", "log10_perm_zz", "layer_id"]

        # for visualization export the perm and layer id
        for _, d in self.gb:
            d[pp.STATE][names[0]] = np.log10(self.perm[:, 0])
            d[pp.STATE][names[1]] = np.log10(self.perm[:, 1])
            d[pp.STATE][names[2]] = np.log10(self.perm[:, 2])
            d[pp.STATE][names[3]] = self.layers_id

        return names

# ------------------------------------------------------------------------------#

    def perm_as_dict(self):
        return {"kxx": self.perm[:, 0], "kyy": self.perm[:, 1], "kzz": self.perm[:, 2]}

# ------------------------------------------------------------------------------#
