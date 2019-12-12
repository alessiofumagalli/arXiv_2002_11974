import numpy as np

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
