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
from simplexify import simplexify
from decoarsify import decoarsify

# ------------------------------------------------------------------------------#

def main(name, gb, coarse=False):

    tol = 1e-6
    case = "case1"

    if coarse:
        partition = pp.coarsening.create_aggregations(gb, weight=0.5)
        partition = pp.coarsening.reorder_partition(partition)
        pp.coarsening.generate_coarse_grid(gb, partition)

    # the flow problem
    param = {
        "tol": tol,
        "k": 1,
        "aperture": 1e-4,
    }
    set_flag(gb)
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
        d[pp.STATE]["cell_volumes"] = g.cell_volumes

    save_vars += ["norm_A", "norm_S", "ratio", "cell_volumes"]

    # output the solution
    save = pp.Exporter(gb, case, folder=folder, binary=False)
    save.write_vtk(save_vars)

    if coarse:
        gb = decoarsify(gb, partition, save_vars)

    gb = simplexify(gb, save_vars)

    # output the solution
    save = pp.Exporter(gb, case, folder=folder+"_simplexify", binary=False)
    save.write_vtk(save_vars)


# ------------------------------------------------------------------------------#

if __name__ == "__main__":

    geometry = "../../geometry/"

    ## ---- Simplex grid ---- #
    mesh_size = np.power(2., -4)
    file_name = geometry + "case1_network.csv"
    gb = create_gb(file_name, mesh_size)
    main("delaunay", gb, coarse=False)

    ## ---- Simplex coarsened grid ---- #
    gb = create_gb(file_name, mesh_size)
    main("delaunay_coarse", gb, coarse=True)


    ## ---- Cartesian cut grid ---- #
    folder = geometry + "case1_cut/"
    gb = import_gb(folder, 2)
    main("cut", gb, coarse=False)

    ## ---- Cartesian coarsened cut grid ---- #
    gb = import_gb(folder, 2)
    main("cut_coarse", gb, coarse=True)

    # ---- Voronoi grid ---- #
    folder = geometry + "case1_voronoi/"
    gb = import_gb(folder, 2)
    main("voronoi", gb, coarse=False)

    # ---- Voronoi grid ---- #
    gb = import_gb(folder, 2)
    main("voronoi_coarse", gb, coarse=True)
