import numpy as np
import scipy.sparse as sps
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from discretization import Flow

from spe10 import Spe10

# ------------------------------------------------------------------------------#

def bc_flag(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    coord_min, coord_max = g.bounding_box()

    # define outflow type boundary conditions
    out_flow = b_face_centers[1] > coord_max[1] - tol

    # define inflow type boundary conditions
    in_flow = b_face_centers[1] < coord_min[1] + tol

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    labels[in_flow + out_flow] = "dir"
    bc_val[b_faces[in_flow]] = 0
    bc_val[b_faces[out_flow]] = 1

    return labels, bc_val

# ------------------------------------------------------------------------------#

def source(cell_centers):
    return np.zeros(cell_centers.shape[1])

# ------------------------------------------------------------------------------#


def main():

    tol = 1e-6
    case = "case3"

    selected_layers = np.arange(1) #np.arange(85)
    spe10 = Spe10(selected_layers)

    perm_folder = "../../geometry/spe10/perm/"
    spe10.read_perm(perm_folder)

    # NOTE: the coarsen implementation is quite inefficient, used only to make a point
    spe10.coarsen()

    # the flow problem
    param = {"tol": tol}
    param.update(spe10.perm_as_dict())

    # exporter
    save = pp.Exporter(spe10.gb, case, folder="solution")
    save_vars = ["pressure", "P0_darcy_flux"]

    # -- flow -- #
    flow = Flow(spe10.gb)
    flow.set_data(param, bc_flag, source)

    # create the matrix for the Darcy problem
    A, b = flow.matrix_rhs()

    # solve the problem
    x = sps.linalg.spsolve(A, b)

    # solve the problem
    flow.extract(x)

    # export the variables
    save.write_vtk(save_vars + spe10.save_perm())

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
