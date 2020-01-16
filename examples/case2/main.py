import os
import numpy as np
import scipy.sparse as sps
from scipy.io import mmwrite

import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from flow import Flow
from flow_tpfa import FlowTpfa

from spe10 import Spe10
from data import *

# ------------------------------------------------------------------------------#

def main(selected_layers, what, discr, gb_ref=None, folder_ref=None):

    tol = 1e-6
    case = "case2"

    selected_layers = np.atleast_1d(selected_layers)
    spe10 = Spe10(selected_layers)

    perm_folder = "../../geometry/spe10/perm/"
    spe10.read_perm(perm_folder)
    save_vars = ["pressure", "P0_darcy_flux"]

    # NOTE: the coarsen implementation is quite inefficient, used only to make a point
    if "mean" in what:
        spe10.coarsen(cdepth=2, epsilon=0.25, mean=what)

    # the flow problem
    param = {"tol": tol, "aperture": 1}
    param.update(spe10.perm_as_dict())

    # exporter
    folder = what + "_" + np.array2string(selected_layers, separator="_")[1:-1]
    if not os.path.exists(folder):
        os.makedirs(folder)

    # -- flow -- #
    flow = discr(spe10.gb, folder)
    flow.set_data(param, bc_flag, source)

    # create the matrix for the Darcy problem
    A, b = flow.matrix_rhs()

    # solve the problem
    x = sps.linalg.spsolve(A, b)

    # solve the problem
    flow.extract(x)

    # export the matrix
    mmwrite(folder + "/matrix.mtx", A)

    # export extra data
    with open(folder + "/data.txt", "w") as f:
        f.write("num_faces " + str(spe10.gb.num_faces()) + "\n")
        f.write("num_cells " + str(spe10.gb.num_cells()) + "\n")
        f.write("diam " + str(spe10.gb.diameter()) + "\n")

    # export the pressure on the original mesh
    p = spe10.map_back("pressure")
    np.savetxt(folder + "/pressure.txt", p)

    # export the cell volumes of the original mesh
    cell_volumes = spe10.map_back("cell_volumes")
    np.savetxt(folder + "/cell_volumes.txt", cell_volumes)

    # save the file with the stabilization
    if discr is Flow:
        norm_A, norm_S, ratio = np.loadtxt(folder + "/stabilization.csv", delimiter=",").T
        for g, d in spe10.gb:
            d[pp.STATE]["norm_A"] = norm_A
            d[pp.STATE]["norm_S"] = norm_S
            d[pp.STATE]["ratio"] = ratio
        save_vars += ["norm_A", "norm_S", "ratio"]

    # compute the error between the reference solution and the current one
    if folder_ref is not None and gb_ref is not None:
        p_ref = np.loadtxt(folder_ref + "/pressure.txt")
        for g, d in gb_ref:
            d[pp.STATE]["error"] = np.abs(p - p_ref)
            d[pp.STATE]["error_rel"] = np.abs(p - p_ref) / p_ref
            d[pp.STATE]["error_l2"] = np.power(p - p_ref, 2) * g.cell_volumes
            d[pp.STATE]["error_l2_rel"] = np.power(p - p_ref, 2) * g.cell_volumes / p_ref
        save = pp.Exporter(gb_ref, case+"_error", folder=folder)
        save.write_vtk(["error", "error_l2", "error_rel", "error_l2_rel"])

    # export the variables
    save = pp.Exporter(spe10.gb, case, folder=folder, binary=False)
    save.write_vtk(save_vars + spe10.save_perm())

    return folder, spe10.gb

# ------------------------------------------------------------------------------#

def run_all(selected_layers):
    selected_layers = np.atleast_1d(selected_layers)

    # -- reference solution TPFA -- #
    folder_ref, gb_ref = main(selected_layers, "reference", FlowTpfa)

    # -- coarsening with MVEM -- #
    folder_mvem_mean, _ = main(selected_layers, "mvem_mean", Flow, gb_ref, folder_ref)
    folder_mvem_hmean, _ = main(selected_layers, "mvem_hmean", Flow, gb_ref, folder_ref)

    # -- load data for the post-process -- #
    cell_volumes = np.loadtxt(folder_ref + "/cell_volumes.txt")

    p_ref = np.loadtxt(folder_ref + "/pressure.txt")

    p_mvem_mean = np.loadtxt(folder_mvem_mean + "/pressure.txt")
    p_mvem_hmean = np.loadtxt(folder_mvem_hmean + "/pressure.txt")

    # -- compute the errors -- #
    norm_p_ref = np.sqrt(np.sum(np.power(p_ref, 2) * cell_volumes))

    err_p_mvem_mean = np.sqrt(np.sum(np.power(p_ref - p_mvem_mean, 2) * cell_volumes))/norm_p_ref
    err_p_mvem_hmean = np.sqrt(np.sum(np.power(p_ref - p_mvem_hmean, 2) * cell_volumes))/norm_p_ref

    # export error
    fname = "error_" + np.array2string(selected_layers, separator="_")[1:-1] + ".txt"
    with open(fname, "w") as f:
        f.write("err pressure mvem mean " + str(err_p_mvem_mean) + "\n")
        f.write("err pressure mvem hmean " + str(err_p_mvem_hmean) + "\n")

# ------------------------------------------------------------------------------#

if __name__ == "__main__":

    # -- layer 3 -- #
    run_all(3)

    # -- layer 35 -- #
    run_all(35)
