import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

import porepy as pp

from data import *

import sys; sys.path.insert(0, "../../src/")
from flow import Flow
from import_grid import import_gb

# ------------------------------------------------------------------------------#

def main(name, gb, coarse=False):

    case = "case1"
    if coarse:
        pp.coarsening.coarsen(gb, "by_volume")

    # the flow problem
    param = {
        "tol": tol,
        "k": 1,
        "aperture": 1e-4,
    }
    set_flag(gb, tol)

    # ---- flow ---- #
    flow = Flow(gb)
    flow.set_data(param, bc_flag, source)

    # create the matrix for the Darcy problem
    A, b = flow.matrix_rhs()

    # solve the problem
    x = sps.linalg.spsolve(A, b)

    # solve the problem
    flow.extract(x)

    # output the solution
    save = pp.Exporter(gb, case, folder="solution_"+name)
    save_vars = ["pressure", "P0_darcy_flux"]
    save.write_vtk(save_vars)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    tol = 1e-6
    coarse = True

    # ---- Simplex grid ---- #
    mesh_size = np.power(2., -3)
    file_name = "network_simple.csv"
    gb = create_gb(file_name, mesh_size)

    main("simplex", gb, coarse)

    # ---- Cartesian cut grid ---- #
    folder = "../../geometry/mesh_test_porepy/meshconuna/"
    gb = import_gb(folder, 2, num_frac=1)

    main("cartesian_cut", gb, coarse)
