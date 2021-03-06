import paraview.simple as pv

import csv
import numpy as np
from scipy.io import mmread

#------------------------------------------------------------------------------#

def plot_over_line(file_in, file_out, pts, resolution=2000):

    if file_in.lower().endswith('.pvd'):
        # create a new 'PVD Reader'
        sol = pv.PVDReader(FileName=file_in)
    elif file_in.lower().endswith('.vtu'):
        # create a new 'XML Unstructured Grid Reader'
        sol = pv.XMLUnstructuredGridReader(FileName=file_in)
    else:
        raise ValueError, "file format not yet supported"

    # create a new 'Plot Over Line'
    pol = pv.PlotOverLine(Input=sol, Source='High Resolution Line Source')

    # Properties modified on plotOverLine1.Source
    pol.Source.Point1 = pts[0]
    pol.Source.Point2 = pts[1]
    pol.Source.Resolution = resolution

    # save data
    pv.SaveData(file_out, proxy=pol, Precision=15)

#------------------------------------------------------------------------------#

def read_csv(file_in, fields=None):

    # post-process the file by selecting only few columns
    if fields is not None:
        data = list(list() for _ in fields)
        with open(file_in, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            [d.append(row[f]) for row in reader for f, d in zip(fields, data)]
    else:
        with open(file_in, 'r') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

    return data

#------------------------------------------------------------------------------#

def write_csv(file_out, fields, data):
    with open(file_out, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        #writer.writeheader()
        [writer.writerow({f: d for f, d in zip(fields, dd)}) for dd in zip(*data)]

#------------------------------------------------------------------------------#

if __name__ == "__main__":

    solver_names = ["delaunay", "delaunay_coarse", "cut", "cut_coarse", "voronoi", "voronoi_coarse"]

    for solver in solver_names:
        folder = "./solution_"+solver+"/"

        field = "pressure"
        # file of the matrix
        file_in = folder+"case1_2.vtu"

        file_out = folder+"pol.csv"
        pts = [[0, 0.5, 0], [1, 0.9, 0]]

        plot_over_line(file_in, file_out, pts)
        data = read_csv(file_out, ['arc_length', field])
        write_csv(file_out, ['arc_length', field], data)
