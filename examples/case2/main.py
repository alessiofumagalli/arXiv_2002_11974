import numpy as np
import scipy.sparse as sps
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from flow import Flow

# ------------------------------------------------------------------------------#

def bc_flag(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    b_dir_low = b_face_centers[0, :] > 1 - tol
    b_dir_high = b_face_centers[0, :] < 0 + tol

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    labels[np.logical_or(b_dir_low, b_dir_high)] = "dir"

    bc_val = np.zeros(g.num_faces)

    bc_val[b_faces[b_dir_low]] = 1
    bc_val[b_faces[b_dir_high]] = 4

    return labels, bc_val

# ------------------------------------------------------------------------------#

def source(cell_centers):
    return np.zeros(cell_centers.shape[1])

# ------------------------------------------------------------------------------#

def create_gb(file_name, mesh_size):
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}

    # Generate a mixed-dimensional mesh
    return network.mesh(mesh_kwargs)

# ------------------------------------------------------------------------------#

def main():

    mesh_size = np.power(2., -3)
    tol = 1e-6
    case = "case1"

    # Define a fracture network in 2d
    gb = create_gb("network.csv", mesh_size)
    #pp.coarsening.coarsen(gb, "by_volume")
    gb.set_porepy_keywords()

    # the flow problem
    param = {
        "tol": tol,
        "k": 1,
        "aperture": 1e-2,
        "kf_t": 1e2, "kf_n": 1e2,
    }

    # exporter
    save = pp.Exporter(gb, case, folder="solution")
    save_vars = ["pressure", "P0_darcy_flux"]

    # -- flow -- #
    flow = Flow(gb)
    flow.set_data(param, bc_flag, source)

    # create the matrix for the Darcy problem
    A, b = flow.matrix_rhs()

    # solve the problem
    x = sps.linalg.spsolve(A, b)

    # solve the problem
    flow.extract(x)

    save.write_vtk(save_vars)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
