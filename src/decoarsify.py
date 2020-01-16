import numpy as np
import porepy as pp

def decoarsify(gb, partition, data=[]):
    "make the higher dimensional grid mapped back from coarsening to the original one"
    new_gb = gb.copy()
    grids = gb.grids_of_dimension(gb.dim_max())

    for g in grids:
        new_g = partition[g][0]
        mask = partition[g][1]
        new_gb.update_nodes({g: new_g})

        for d in data:
            if gb._nodes[g][pp.STATE].get(d, None) is not None:
                old_values = gb._nodes[g][pp.STATE][d]
                if old_values.ndim > 1:
                    new_gb._nodes[new_g][pp.STATE][d] = np.zeros((old_values.shape[0], new_g.num_cells))
                    for dim in np.arange(old_values.ndim):
                        new_gb._nodes[new_g][pp.STATE][d][dim, :] = old_values[dim, mask]
                else:
                    new_gb._nodes[new_g][pp.STATE][d] = old_values[mask]
            else:
                old_values = gb._nodes[g][d]
                new_gb._nodes[new_g][d] = old_values[mask]

    return new_gb
