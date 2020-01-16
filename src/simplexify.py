import numpy as np
from scipy import sparse as sps
import porepy as pp

def simplexify(gb, data=[]):
    "make the higher dimensional grid make of simplices by splitting the cells and return the map"
    new_gb = gb.copy()
    grids = gb.grids_of_dimension(gb.dim_max())

    for g in grids:
        g_cell_nodes = g.cell_nodes()
        face_nodes_indices = np.empty(0, dtype=np.int) #g.num_faces)
        cell_faces_indices = np.empty(0, dtype=np.int)
        cell_faces_data = np.empty(0, dtype=np.int)
        # projection operator
        proj_I = np.empty(0, dtype=np.int)
        proj_J = np.empty(0, dtype=np.int)

        num_faces = g.num_faces
        num_cells = 0

        for c in np.arange(g.num_cells):
            # cell nodes
            c_nodes_pos = slice(g_cell_nodes.indptr[c], g_cell_nodes.indptr[c+1])
            c_nodes = g_cell_nodes.indices[c_nodes_pos]

            # cell faces
            c_faces_pos = slice(g.cell_faces.indptr[c], g.cell_faces.indptr[c+1])
            c_faces = g.cell_faces.indices[c_faces_pos]
            c_faces_data = g.cell_faces.data[c_faces_pos]

            out = np.ones((4, c_faces.size), dtype=np.int)
            out[0] = np.arange(c_faces.size) + num_faces
            out[1] = c_nodes
            out[2] = c + g.num_nodes

            for f, d in zip(c_faces, c_faces_data):
                f_nodes_pos = slice(g.face_nodes.indptr[f], g.face_nodes.indptr[f+1])
                f_nodes = g.face_nodes.indices[f_nodes_pos]

                # check which new faces we need to consider
                mask = np.in1d(out[1], f_nodes, assume_unique=True)
                c_indices = np.hstack((out[0, mask], f))
                c_data = np.hstack((out[3, mask], d))
                out[3, mask] *= -1

                # save the data
                cell_faces_indices = np.hstack((cell_faces_indices, c_indices))
                cell_faces_data = np.hstack((cell_faces_data, c_data))

            proj_I = np.hstack((proj_I, num_cells + np.arange(c_faces.size)))
            proj_J = np.hstack((proj_J, c*np.ones(c_faces.size, dtype=np.int)))

            num_cells += c_faces.size
            num_faces += c_faces.size
            face_nodes_indices = np.hstack((face_nodes_indices, out[1:-1].ravel(order='F')))

        face_nodes_indices = np.hstack((g.face_nodes.indices, face_nodes_indices))
        face_nodes_indptr = np.arange(num_faces+1, dtype=np.int)*2
        face_nodes_data = np.ones(face_nodes_indices.size, dtype=np.bool)
        face_nodes = sps.csc_matrix((face_nodes_data, face_nodes_indices, face_nodes_indptr))

        cell_faces_indptr = np.arange(num_cells+1, dtype=np.int)*3
        cell_faces = sps.csc_matrix((cell_faces_data, cell_faces_indices, cell_faces_indptr))

        nodes = np.hstack((g.nodes, g.cell_centers))

        new_g = pp.Grid(g.dim, nodes, face_nodes, cell_faces, g.name)
        new_g.compute_geometry()
        new_gb.update_nodes({g: new_g})

        # update the data
        proj = sps.csr_matrix((np.ones(proj_J.size), (proj_I, proj_J)))
        for d in data:
            if gb._nodes[g][pp.STATE].get(d, None) is not None:
                old_values = gb._nodes[g][pp.STATE][d]
                if old_values.ndim > 1:
                    new_gb._nodes[new_g][pp.STATE][d] = np.zeros((old_values.shape[0], proj.shape[0]))
                    for dim in np.arange(old_values.ndim):
                        new_gb._nodes[new_g][pp.STATE][d][dim, :] = proj*old_values[dim, :]
                else:
                    new_gb._nodes[new_g][pp.STATE][d] = proj*old_values
            else:
                old_values = gb._nodes[g][d]
                new_gb._nodes[new_g][d] = proj*old_values

    return new_gb

