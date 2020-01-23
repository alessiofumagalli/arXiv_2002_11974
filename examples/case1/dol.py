import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------------------------#

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=12)


def plot_single(data, legend, title, col=(0, 1)):

    plt.figure(0)
    plt.plot(data[:, col[0]], data[:, col[1]], label=legend)
    plt.title(title)
    plt.xlabel("arc length")
    plt.ylabel("$p$ difference")
    plt.grid(True)
    plt.legend()


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

    file_ref = folder + "REF_0p50p9LR.csv"
    ref = np.loadtxt(file_ref, delimiter=",", skiprows=1)[:, 0]

    # plot results
    solvers = ["delaunay", "delaunay_coarse", "cut", "cut_coarse", "voronoi", "voronoi_coarse"]
    folder_out = "./img/"

    title = "pressure difference over line"
    for solver in solvers:
        file_name = "./solution_" + solver + "/pol.csv"
        val, diff = np.loadtxt(file_name, delimiter=",").T
        diff = np.abs(diff[::2] - ref) / ref

        data = np.vstack((val[::2], diff)).T
        plot_single(data, solver.replace("_", " "), title)

    # save
    name = "dol"
    save_single(name, folder_out)

# ------------------------------------------------------------------------------#

if __name__ == "__main__":
    main()
