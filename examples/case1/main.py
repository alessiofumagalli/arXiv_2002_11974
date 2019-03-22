import numpy as np
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

def main():

    h = 0.025
    tol = 1e-6
    mesh_args = {"mesh_size_frac": h}
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 2}
    folder = "case1"

    # Define a fracture network in 2d
    file_name = "network.csv"
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    # Generate a mixed-dimensional mesh
    gb = network.mesh(mesh_args)

    # the flow problem
    param = {
        "domain": gb.bounding_box(as_dict=True),
        "tol": tol,
        "k": 1,
        "aperture": 1e-2, "kf_t": 1e2, "kf_n": 1e2,
    }

    # declare the flow problem and the multiscale solver
    flow = Flow(gb, folder, tol)

    # set the data
    flow.data(param, bc_flag)

    # create the matrix for the Darcy problem
    A, b, block_dof, full_dof = flow.matrix_rhs()

    # solve the problem
    x = flow.solve(A, b)

    # solve the problem
    flow.extract(x, block_dof, full_dof)
    flow.export()

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
