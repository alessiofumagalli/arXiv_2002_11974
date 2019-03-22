import scipy.sparse as sps
import numpy as np
import porepy as pp

from logger import logger

# ------------------------------------------------------------------------------#

class Flow(object):

    def __init__(self, gb, folder, tol):

        self.model = "flow"
        self.gb = gb

        # discretization operator name
        self.discr_name = "flux"
        self.discr = pp.MVEM(self.model)

        self.coupling_name = self.discr_name + "_coupling"
        self.coupling = pp.RobinCoupling(self.model, self.discr)

        self.source_name = "source"
        self.source = pp.DualScalarSource(self.model)

        # master variable name
        self.variable = "flow_variable"
        self.mortar = "lambda_" + self.variable

        # post process variables
        self.pressure = "pressure"
        self.flux = "darcy_flux"  # it has to be this one
        self.P0_flux = "P0_darcy_flux"

        # tolerance
        self.tol = tol

        # exporter
        self.save = pp.Exporter(self.gb, "solution", folder=folder)

    def data(self, data, bc_flag):

        for g, d in self.gb:
            param = {}

            unity = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)

            d["is_tangential"] = True
            d["tol"] = self.tol

            # assign permeability
            if g.dim < self.gb.dim_max():
                kxx = data["kf_t"] * unity
                perm = pp.SecondOrderTensor(1, kxx=kxx, kyy=1, kzz=1)
                aperture = data["aperture"] * unity

            else:
                kxx = data["k"] * unity
                if g.dim == 2:
                    perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
                else:
                    perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=kxx)
                aperture = unity

            param["second_order_tensor"] = perm
            param["aperture"] = aperture

            param["source"] = g.cell_volumes * (g.cell_centers[1, :] < 0.5+self.tol)

            # Boundaries
            b_faces = g.tags["domain_boundary_faces"].nonzero()[0]
            if b_faces.size:
                labels, bc_val = bc_flag(g, data, self.tol)
                param["bc"] = pp.BoundaryCondition(g, b_faces, labels)
            else:
                bc_val = np.zeros(g.num_faces)
                param["bc"] = pp.BoundaryCondition(g, empty, empty)

            param["bc_values"] = bc_val

            d[pp.PARAMETERS] = pp.Parameters(g, self.model, param)

        for e, d in self.gb.edges():
            g_l = self.gb.nodes_of_edge(e)[0]

            mg = d["mortar_grid"]
            check_P = mg.slave_to_mortar_avg()

            aperture = self.gb.node_props(g_l, pp.PARAMETERS)[self.model]["aperture"]
            gamma = check_P * aperture
            kn = data["kf_n"] * np.ones(mg.num_cells) / gamma
            param = {"normal_diffusivity": kn}

            d[pp.PARAMETERS] = pp.Parameters(e, self.model, param)

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
                    e: (self.mortar, self.coupling),
                }
            }

    # ------------------------------------------------------------------------------#

    def matrix_rhs(self):

        # empty the matrices
        for g, d in self.gb:
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        for e, d in self.gb.edges():
            d[pp.DISCRETIZATION_MATRICES] = {self.model: {}}

        # solution of the darcy problem
        assembler = pp.Assembler()

        logger.info("Assemble the flow problem")
        block_A, block_b, block_dof, full_dof = assembler.assemble_matrix_rhs(self.gb,
            active_variables=[self.variable, self.mortar], add_matrices=False)

        # unpack the matrices just computed
        coupling_name = self.coupling_name + (
            "_" + self.mortar + "_" + self.variable + "_" + self.variable
        )
        discr_name = self.discr_name + "_" + self.variable
        source_name = self.source_name + "_" + self.variable

        # need a sign for the convention of the conservation equation
        A = block_A[discr_name]
        b = block_b[discr_name] + block_b[source_name]

        if coupling_name in block_A:
            A += block_A[coupling_name]
            b += block_b[coupling_name]

        return A, b, block_dof, full_dof

    # ------------------------------------------------------------------------------#

    def solve(self, A, b):

        logger.info("Solve the linear system")
        x = sps.linalg.spsolve(A, b)
        logger.info("done")

        return x

    # ------------------------------------------------------------------------------#

    def extract(self, x, block_dof, full_dof):

        logger.info("Variable post-process")
        assembler = pp.Assembler()
        assembler.distribute_variable(self.gb, x, block_dof, full_dof)
        for g, d in self.gb:
            d[self.pressure] = self.discr.extract_pressure(g, d[self.variable], d)
            d[self.flux] = self.discr.extract_flux(g, d[self.variable], d)

        # export the P0 flux reconstruction
        pp.project_flux(self.gb, self.discr, self.flux, self.P0_flux, self.mortar)
        logger.info("done")

    # ------------------------------------------------------------------------------#

    def export(self):
        logger.info("Export variables")
        self.save.write_vtk([self.pressure, self.P0_flux])
        logger.info("done")

    # ------------------------------------------------------------------------------#
