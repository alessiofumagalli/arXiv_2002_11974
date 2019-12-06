import glob
import os
import numpy as np
from scipy import sparse as sps

import porepy as pp

def import_grid(folder, fname, dim, **kwargs):

    index_shift = kwargs.get("index_shift", 1)

    # load the nodes
    fnodes_root = kwargs.get("nodes", "points")
    fnodes = folder + "/" + fnodes_root + fname + ".txt"
    nodes = np.loadtxt(fnodes, dtype=np.float).T
    nodes = np.vstack([nodes[1:], np.zeros(nodes.shape[1])])

    # load the face to node map
    fface_nodes_root = kwargs.get("face_nodes", "face2node")
    fface_nodes = folder + "/" + fface_nodes_root + fname + ".txt"
    face_nodes = np.loadtxt(fface_nodes, dtype=np.int).T

    col = face_nodes[0] - index_shift
    row = face_nodes[1] - index_shift
    data = face_nodes[2].astype(np.bool)
    face_nodes = sps.csc_matrix((data, (row, col)))

    # load the cell to face map
    fcell_faces_root = kwargs.get("cell_faces", "cell2face")
    fcell_faces = folder + "/" + fcell_faces_root + fname + ".txt"
    cell_faces = np.loadtxt(fcell_faces, dtype=np.int).T

    col = cell_faces[0] - index_shift
    row = cell_faces[1] - index_shift
    data = cell_faces[2]
    cell_faces = sps.csc_matrix((data, (row, col)))

    # load the volume
    #fcell_volumes_root = kwargs.get("cell_volumes", "volumes")
    #fcell_volumes = folder + "/" + fcell_volumes_root + fname + ".txt"
    #cell_volumes = np.loadtxt(fcell_volumes, dtype=np.float).T

    # I CENTRI DELLE CELLE DEVONO ESSERE CALCOLATI IN MANIERA SPECIALE

    # it's not really triangular grid but it's useful somewhere
    name = "TriangleGrid" if dim == 2 else "TensorGrid"
    g = pp.Grid(dim, nodes, face_nodes, cell_faces, name)
    g.compute_geometry()

    # load the cell to local to global map
    fglobal_root = kwargs.get("global", "global")
    fglobal = folder + "/" + fglobal_root + fname + ".txt"
    g.global_point_ind = np.loadtxt(fglobal, dtype=np.int).T[1]

    return g

def import_gb(folder, max_dim, num_frac, **kwargs):

    index_names = kwargs.get("index_names", 1)
    index_shift = kwargs.get("index_shift", 1)

    # bulk grid
    fname_bulk = kwargs.get("bulk", "_bulk")
    grid_bulk = [import_grid(folder, fname_bulk, max_dim)]

    # fracture grids
    fname_fracture = kwargs.get("fracture", "_fracture_")
    grid_fracture = []
    for idx in np.arange(num_frac):
        fname = fname_fracture + str(idx+index_names)
        g = import_grid(folder, fname, max_dim-1)
        grid_fracture.append(g)

    grids = [grid_bulk, grid_fracture, []]
    gb = pp.meshing.grid_list_to_grid_bucket(grids)

    return gb
