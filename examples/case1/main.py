import numpy as np
import scipy.sparse as sps
import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from discretization import Flow

# ------------------------------------------------------------------------------#

def bc_flag(g, data, tol):
    b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
    b_face_centers = g.face_centers[:, b_faces]

    # define outflow type boundary conditions
    out_flow = b_face_centers[1] > 2 - tol

    # define inflow type boundary conditions
    in_flow = b_face_centers[1] < 0 + tol

    # define the labels and values for the boundary faces
    labels = np.array(["neu"] * b_faces.size)
    bc_val = np.zeros(g.num_faces)

    if g.dim == 2:
        labels[in_flow + out_flow] = "dir"
        bc_val[b_faces[in_flow]] = 0
        bc_val[b_faces[out_flow]] = 1
    else:
        labels[:] = "dir"
        bc_val[b_faces] = (b_face_centers[0, :] < 0.5).astype(np.float)

    return labels, bc_val

# ------------------------------------------------------------------------------#

def source(cell_centers):
    return np.zeros(cell_centers.shape[1])

# ------------------------------------------------------------------------------#

def create_gb(file_name, mesh_size):
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 2}
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
