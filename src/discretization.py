import scipy.sparse as sps
import numpy as np
import porepy as pp

from my_mvem import My_MVEM

class Flow(object):

    # ------------------------------------------------------------------------------#

    def __init__(self, gb, model="flow"):

        self.model = model
        self.gb = gb
        self.data = None
        self.assembler = None

        # discretization operator name
        self.discr_name = self.model + "_flux"
        self.discr = My_MVEM(self.model)

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

        for g, d in self.gb:
            param = {}

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            d["is_tangential"] = True
            d["tol"] = data["tol"]

            # assign permeability
            if g.dim < self.gb.dim_max():
                kxx = data["kf_t"] * unity
                perm = pp.SecondOrderTensor(1, kxx=kxx, kyy=1, kzz=1)
                aperture = data["aperture"] * unity

            else:
                # check if the permeability is isotropic or not
                k = data.get("k", None)
                if k is not None:
                    kxx, kyy, kzz = k * unity, k * unity, k * unity
                else:
                    kxx, kyy, kzz = data["kxx"], data["kyy"], data["kzz"]
                if g.dim == 2:
                    perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kyy, kzz=1)
                else:
                    perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kyy, kzz=kzz)
                aperture = unity


            param["second_order_tensor"] = perm
            param["aperture"] = aperture

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

            d[pp.PARAMETERS].update(pp.Parameters(g, self.model, param))

        for e, d in self.gb.edges():
            g_l = self.gb.nodes_of_edge(e)[0]

            mg = d["mortar_grid"]
            check_P = mg.slave_to_mortar_avg()

            aperture = self.gb.node_props(g_l, pp.PARAMETERS)[self.model]["aperture"]
            gamma = check_P * aperture
            kn = data["kf_n"] * np.ones(mg.num_cells) / gamma
            param = {"normal_diffusivity": kn}

            d[pp.PARAMETERS].update(pp.Parameters(e, self.model, param))

        # set now the discretization

        # set the discretization for the grids
        for g, d in self.gb:
            d[pp.PRIMARY_VARIABLES].update({self.variable: {"cells": 1, "faces": 1}})
            d[pp.DISCRETIZATION].update({self.variable: {self.discr_name: self.discr,
                                                         self.source_name: self.source}})

        # define the interface terms to couple the grids
        for e, d in self.gb.edges():
            g_slave, g_master = self.gb.nodes_of_edge(e)
            d[pp.PRIMARY_VARIABLES].update({self.mortar: {"cells": 1}})
            d[pp.COUPLING_DISCRETIZATION].update({
                self.coupling_name: {
                    g_slave: (self.variable, self.discr_name),
                    g_master: (self.variable, self.discr_name),
                    e: (self.mortar, self.coupling),
                }
            })

        # assembler
        variables = [self.variable, self.mortar]
        self.assembler = pp.Assembler(self.gb, active_variables=variables)

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # empty the matrices
        for g, d in self.gb:
            d[pp.DISCRETIZATION_MATRICES].update({self.model: {}})

        for e, d in self.gb.edges():
            d[pp.DISCRETIZATION_MATRICES].update({self.model: {}})

        return self.assembler.assemble_matrix_rhs()

    # ------------------------------------------------------------------------------#

    def extract(self, x):
        self.assembler.distribute_variable(x)
        for g, d in self.gb:
            var = d[pp.STATE][self.variable]
            d[pp.STATE][self.pressure] = self.discr.extract_pressure(g, var, d)
            d[pp.STATE][self.flux] = self.discr.extract_flux(g, var, d)

        # export the P0 flux reconstruction
        pp.project_flux(self.gb, self.discr, self.flux, self.P0_flux, self.mortar)

    # ------------------------------------------------------------------------------#
