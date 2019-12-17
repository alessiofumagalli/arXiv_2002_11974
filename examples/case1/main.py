import os
import numpy as np
import scipy.sparse as sps
from scipy.io import mmwrite
import matplotlib.pyplot as plt

import porepy as pp

from data import *

import sys; sys.path.insert(0, "../../src/")
from flow import Flow
from import_grid import import_gb

# ------------------------------------------------------------------------------#

def main(name, gb, coarse=False):

    tol = 1e-6
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
    save_vars = ["pressure", "P0_darcy_flux"]
    folder = "solution_" + name
    if not os.path.exists(folder):
        os.makedirs(folder)

    # ---- flow ---- #
    flow = Flow(gb, folder)
    flow.set_data(param, bc_flag, source)

    # create the matrix for the Darcy problem
    A, b = flow.matrix_rhs()

    # solve the problem
    x = sps.linalg.spsolve(A, b)

    flow.extract(x)

    # export the matrix
    mmwrite(folder + "/matrix.mtx", A)

    # export extra data
    with open(folder + "/data.txt", "w") as f:
        f.write("num_faces " + str(gb.num_faces()) + "\n")
        f.write("num_cells " + str(gb.num_cells()) + "\n")
        f.write("diam " + str(gb.diameter()) + "\n")

    # save the file with the stabilization
    # export the stabilization terms only for the matrix
    for g, d in gb:
        if g.dim == 2:
            norm_A, norm_S, ratio = np.loadtxt(folder + "/stabilization.csv", delimiter=",").T
            d[pp.STATE]["norm_A"] = norm_A[:g.num_cells]
            d[pp.STATE]["norm_S"] = norm_S[:g.num_cells]
            d[pp.STATE]["ratio"] = ratio[:g.num_cells]
        else:
            d[pp.STATE]["norm_A"] = np.zeros(g.num_cells)
            d[pp.STATE]["norm_S"] = np.zeros(g.num_cells)
            d[pp.STATE]["ratio"] = np.zeros(g.num_cells)

    save_vars += ["norm_A", "norm_S", "ratio"]

    # output the solution
    save = pp.Exporter(gb, case, folder=folder, binary=False)
    save.write_vtk(save_vars)

# ------------------------------------------------------------------------------#

def run_all(refinement, coarse):

        # ---- Simplex grid ---- #
        #mesh_size = np.power(2., -3)
        #file_name = "network_simple.csv"
        #gb = create_gb(file_name, mesh_size)

        #main("simplex", gb, coarse)

        # ---- Cartesian cut grid ---- #
        folder = "../../geometry/mesh_test_porepy/meshcondue_new/"
        gb = import_gb(folder, 2)

        main("cartesian_due", gb, coarse)

        # ---- Voronoi ---- #

        # ---- Simplified simplex grid ---- #


# ------------------------------------------------------------------------------#

if __name__ == "__main__":

    # num cells 1300, 4700, 18000
    refinements = [0] #, 1, 2]

    for refinement in refinements:
        #run_all(refinement, coarse=False)
        run_all(refinement, coarse=True)
