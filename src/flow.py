import numpy as np
import porepy as pp

from my_mvem import My_MVEM

class Flow(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb, folder, model="flow"):

        self.model = model
        self.gb = gb
        self.data = None
        self.assembler = None

        # discretization operator name
        self.discr_name = self.model + "_flux"
        self.discr = My_MVEM(self.model, folder)

        self.coupling_name = self.discr_name + "_coupling"
        self.coupling = pp.RobinCoupling(self.model, self.discr)

        self.source_name = self.model + "_source"
        self.source = pp.DualScalarSource(self.model)

        # master variable name
        self.variable = self.model + "_variable"
        self.mortar = self.model + "_lambda"

        # post process variables
        self.pressure = "pressure"
        self.flux = "darcy_flux"  # it has to be this one
        self.P0_flux = "P0_darcy_flux"

    # ------------------------------------------------------------------------------#

    def set_data(self, data, bc_flag, source):
        self.data = data

        for g, d in self.gb.nodes():

            param = {}

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            d["is_tangential"] = True
            d["tol"] = data["tol"]
            aperture = np.power(data["aperture"], self.gb.dim_max() - g.dim) * unity

            # assign permeability
            if g.dim < self.gb.dim_max():
                k_t = d.get("k_t", None)
                if k_t is None:
                    k_t = data["k_t"]
                perm = pp.SecondOrderTensor(kxx=k_t * aperture, kyy=1, kzz=1)
            else:
                # check if the permeability is isotropic or not
                k = data.get("k", None)
                if k is not None:
                    kxx, kyy, kzz = [k * unity] * 3
                else:
                    kxx, kyy, kzz = data["kxx"] * unity, data["kyy"] * unity, data["kzz"] * unity
                if g.dim == 2:
                    perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kzz=1)
                else:
                    perm = pp.SecondOrderTensor(kxx=kxx, kyy=kyy, kzz=kzz)


            param["second_order_tensor"] = perm
            param["source"] = g.cell_volumes * aperture * source(g.cell_centers)

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size:
                labels, bc_val = bc_flag(g, data, data["tol"])
                param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
            else:
                bc_val = np.zeros(g.num_faces)
                param["bc"] = pp.BoundaryCondition(g, empty, empty)

            param["bc_values"] = bc_val
            pp.initialize_data(g, d, self.model, param)


        for e, d in self.gb.edges():
            mg = d["mortar_grid"]
            _, gh = self.gb.nodes_of_edge(e)

            k_n = d.get("k_n", None)
            if k_n is None:
                k_n = data["k_n"]

            aperture_h = np.power(data["aperture"], self.gb.dim_max() - gh.dim)
            param = {"normal_diffusivity": k_n * 2 / data["aperture"] * aperture_h}
            pp.initialize_data(mg, d, self.model, param)

        # set now the discretization

        # set the discretization for the grids
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES] = {self.variable: {"cells": 1, "faces": 1}}
            d[pp.DISCRETIZATION] = {self.variable: {self.discr_name: self.discr,
                                                    self.source_name: self.source}}

        # define the interface terms to couple the grids
        for e, d in self.gb.edges():
            g_slave, g_master = self.gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES] = {self.mortar: {"cells": 1}}
            d[pp.COUPLING_DISCRETIZATION] = {
                self.coupling_name: {
                    g_slave: (self.variable, self.discr_name),
                    g_master: (self.variable, self.discr_name),
                    e: (self.mortar, self.coupling)}}

        # assembler
        variables = [self.variable, self.mortar]
        self.assembler = pp.Assembler(self.gb, active_variables=variables)

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # empty the matrices
        for g, d in self.gb:
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        for e, d in self.gb.edges():
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        self.assembler.discretize()
        return self.assembler.assemble_matrix_rhs()

    # ------------------------------------------------------------------------------#

    def extract(self, x):
        self.assembler.distribute_variable(x)
        for g, d in self.gb:
            var = d[pp.STATE][self.variable]
            d[pp.STATE][self.pressure] = self.discr.extract_pressure(g, var, d)
            d[pp.STATE][self.flux] = self.discr.extract_flux(g, var, d)
            d[pp.STATE]["cell_volumes"] = g.cell_volumes

        # export the P0 flux reconstruction
        pp.project_flux(self.gb, self.discr, self.flux, self.P0_flux, self.mortar)

    # ------------------------------------------------------------------------------#
