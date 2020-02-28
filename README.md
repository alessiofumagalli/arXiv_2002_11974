# Performances of the mixed virtual element method on complex grids for underground flow

Source code and examples for the paper<br>
"*Performances of the mixed virtual element method on complex grids for underground flow*" by Alessio Fumagalli, Anna Scotti, Luca Formaggia. See [arXiv pre-print](https://arxiv.org/abs/2002.11974).


# Reproduce results from paper
Runscripts for all test cases of the work available [here](./examples).<br>
Note that you may have to revert to an older version of [PorePy](https://github.com/pmgbergen/porepy) to run the examples.

# Abstract
The numerical solution of physical processes in the underground frequently leads to
challenges related to the geometry and/or data. The former is mainly due to the form of
sedimentary layers and the presence of fractures and faults, while the latter is connected
to the properties of the rock matrix which might vary abruptly from rock to rock. The
development, of approximation schemes, is frequently focused on the overcoming of such
difficulties, keeping good properties of the numerical solution. In this work, we carry out a
numerical study for the performances of the numerical scheme called mixed virtual element
method, for the solution of a single-phase flow model. This method is able to handle grid
cells of arbitrary type in all the physical dimensions and it has been proven to be robust
with respect to the variation of the permeability field. Our numerical findings are supported
by two test cases that cover several critical aspects.

# Citing
If you use this work in your research, we ask you to cite the following publication [arXiv:2002.11974 [math.NA]](https://arxiv.org/abs/2002.11974).

# PorePy version
If you want to run the code you need to install [PorePy](https://github.com/pmgbergen/porepy) and revert to commit f82a91c1b6ef83b6954ecd26723f8b7443880891 FIX GIT COMMIT <br>
Newer versions of PorePy may not be compatible with this repository.

# License
See [license](./LICENSE).
