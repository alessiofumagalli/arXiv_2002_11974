import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt

import porepy as pp

# ------------------------------------------------------------------------------#

def create_gb(file_name, mesh_size):
    domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1}
    network = pp.fracture_importer.network_2d_from_csv(file_name, domain=domain)

    mesh_kwargs = {"mesh_size_frac": mesh_size, "mesh_size_min": mesh_size / 20}
    return network.mesh(mesh_kwargs)

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

def set_flag(gb):
    tol = 1e-3
    # set the key for the low peremable fractures
    gb.add_node_props("is_low", "frac_num")
    for g, d in gb:
        d["is_low"] = False
        d["frac_num"] = -1
        if g.dim == 1:

            f_0 = (g.nodes[0, :] - 0.05)/(0.2200 - 0.05) - (g.nodes[1, :] - 0.4160)/(0.0624 - 0.4160)
            if np.sum(np.abs(f_0)) < tol:
                d["frac_num"] = 0
                d["k_t"] = 1e4

            f_1 = (g.nodes[0, :] - 0.05)/(0.2500 - 0.05) - (g.nodes[1, :] - 0.2750)/(0.1350 - 0.2750)
            if np.sum(np.abs(f_1)) < tol:
                d["frac_num"] = 1
                d["k_t"] = 1e4

            f_2 = (g.nodes[0, :] - 0.15)/(0.4500 - 0.15) - (g.nodes[1, :] - 0.6300)/(0.0900 - 0.6300)
            if np.sum(np.abs(f_2)) < tol:
                d["frac_num"] = 2
                d["k_t"] = 1e4

            f_3 = (g.nodes[0, :] - 0.15)/(0.4 - 0.15) - (g.nodes[1, :] - 0.9167)/(0.5 - 0.9167)
            if np.sum(np.abs(f_3)) < tol:
                d["frac_num"] = 3
                d["is_low"] = True
                d["k_t"] = 1e-4

            f_4 = (g.nodes[0, :] - 0.65)/(0.849723 - 0.65) - (g.nodes[1, :] - 0.8333)/(0.167625 - 0.8333)
            if np.sum(np.abs(f_4)) < tol:
                d["frac_num"] = 4
                d["is_low"] = True
                d["k_t"] = 1e-4

            f_5 = (g.nodes[0, :] - 0.70)/(0.849723 - 0.70) - (g.nodes[1, :] - 0.2350)/(0.167625 - 0.2350)
            if np.sum(np.abs(f_5)) < tol:
                d["frac_num"] = 5
                d["k_t"] = 1e4

            f_6 = (g.nodes[0, :] - 0.60)/(0.8500 - 0.60) - (g.nodes[1, :] - 0.3800)/(0.2675 - 0.3800)
            if np.sum(np.abs(f_6)) < tol:
                d["frac_num"] = 6
                d["k_t"] = 1e4

            f_7 = (g.nodes[0, :] - 0.35)/(0.8000 - 0.35) - (g.nodes[1, :] - 0.9714)/(0.7143 - 0.9714)
            if np.sum(np.abs(f_7)) < tol:
                d["frac_num"] = 7
                d["k_t"] = 1e4

            f_8 = (g.nodes[0, :] - 0.75)/(0.9500 - 0.75) - (g.nodes[1, :] - 0.9574)/(0.8155 - 0.9574)
            if np.sum(np.abs(f_8)) < tol:
                d["frac_num"] = 8
                d["k_t"] = 1e4

            f_9 = (g.nodes[0, :] - 0.15)/(0.4000 - 0.15) - (g.nodes[1, :] - 0.8363)/(0.9727 - 0.8363)
            if np.sum(np.abs(f_9)) < tol:
                d["frac_num"] = 9
                d["k_t"] = 1e4
        else:
            d["k_t"] = 1


    # we set know also the flag for the intersection, we need to go first through the
    # 0-dim grids and set there the is low and after to the edges
    gb.add_edge_props("is_low")
    for _, d in gb.edges():
        d["is_low"] = False
        d["k_n"] = 1e4

    for e, d in gb.edges():
        gl, gh = gb.nodes_of_edge(e)
        if gl.dim == 1 and gb.node_props(gl, "is_low"):
            d["k_n"] = 1e-4
        if gl.dim == 0 and gb.node_props(gh, "is_low"):
            gb.set_node_prop(gl, "is_low", True)

    # modify the key only for certain fractures
    for e, d in gb.edges():
        gl, gh = gb.nodes_of_edge(e)
        if gl.dim == 0 and gb.node_props(gl, "is_low"):
            d["k_n"] = 2./(1./1e-4+1./1e4)
            d["is_low"] = True

# ------------------------------------------------------------------------------#
