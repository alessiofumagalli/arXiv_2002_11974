import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------------#

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=12)


def plot_single(file_name, legend, title, col=(0, 1)):

    data = np.loadtxt(file_name, delimiter=",")

    plt.figure(0)
    plt.plot(data[:, col[0]], data[:, col[1]], label=legend)
    plt.title(title)
    plt.xlabel("arc length")
    plt.ylabel("$p$")
    plt.grid(True)
    plt.legend()


# ------------------------------------------------------------------------------#

def plot_known(file_name, arc, col, shift, color="gray", alpha=0.3, label=None):

    data = np.loadtxt(file_name, delimiter=",", skiprows=1)

    plt.figure(0)
    if arc is None:
        arc_length = np.sqrt(np.power(data[:, col[0]] - shift[0], 2) + \
                             np.power(data[:, col[1]] - shift[1], 2))
    else:
        arc_length = data[:, arc]

    plt.plot(arc_length, data[:, col[2]], label=label, color=color, alpha=alpha)

# ------------------------------------------------------------------------------#

def save_single(filename, folder, figure_id=0, extension=".pdf"):

    if not os.path.exists(folder):
        os.makedirs(folder)

    plt.figure(figure_id)
    plt.savefig(folder + filename + extension, bbox_inches="tight")
    plt.gcf().clear()


# ------------------------------------------------------------------------------#


def main():

    # plot known results from the benchmark
    git_repo = "/home/elle/Dropbox/Work/PresentazioniArticoli/2016/Articles/2015-fractures-comparison/"
    folder = git_repo + "/stuttgart-repo/fracture-flow/complex/results/plotLine/"

    knowns = {"REF_0p50p9LR": [None, (3, 4, 0), (0, 0.5)],
              "box_0p50p9LR": [14, (-1, -1, 3), None],
              "tpfa_lr": [2, (-1, -1, 0), None],
              "tpfa_ic_lr": [2, (-1, -1, 0), None],
              "mpfa_lr": [2, (-1, -1, 0), None],
              "edfm_0p50p9LR": [None, (1, 2, 0), (0, 0.5)],
              "mortar_0p50p9LR": [None, (3, 4, 0), (0, 0.5)],
              "DXFEM_0p50p9LR": [None, (1, 2, 0), (0, 0.5)]}

    # plot results
    solvers = ["delaunay", "delaunay_coarse", "cut", "cut_coarse", "voronoi", "voronoi_coarse"]
    folder_out = "./img/"

    title = "pressure over line"
    for known, col_shift in knowns.items():
        data = folder + known + ".csv"
        if known == "REF_0p50p9" or known == "REF_0p50p9LR":
            plot_known(data, *col_shift, color="black", alpha=0.7, label="reference")
        else:
            plot_known(data, *col_shift)

    for solver in solvers:
        folder_in = "./solution_" + solver + "/"
        data = folder_in + "pol.csv"
        plot_single(data, solver.replace("_", " "), title)

    # save
    name = "pol"
    save_single(name, folder_out)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
