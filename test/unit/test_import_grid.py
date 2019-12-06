
import numpy as np
import unittest

import porepy as pp

import sys; sys.path.insert(0, "../../src/")
from import_grid import import_grid, import_gb

# ------------------------------------------------------------------------------#

class TestImportGrid(unittest.TestCase):

    def test_import_single_grid(self):
        folder = "../../geometry/mesh_test_porepy/mesh_senzafrattura/"
        fname = "_bulk"
        g = import_grid(folder, fname, 2)
        pp.Exporter(g, "grid").write_vtk()

        pp.plot_grid(g, info="c", alpha=0)
        print(g)

    def test_import_single_gb(self):
        folder = "../../geometry/mesh_test_porepy/meshconuna/"
        gb = import_gb(folder, 2, num_frac=1)
        for g, d in gb:
            d[pp.STATE] = {}

        pp.Exporter(gb, "grid").write_vtk()

    def test_import_gb_2(self):
        folder = "../../geometry/mesh_test_porepy/meshcondue/"
        gb = import_gb(folder, 2, num_frac=2)
        pp.Exporter(gb, "grid").write_vtk()
        for g, d in gb:
            d[pp.STATE] = {}

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    TestImportGrid().test_import_gb_2()
    #unittest.main()
