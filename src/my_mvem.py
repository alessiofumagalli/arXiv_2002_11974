import numpy as np
import porepy as pp

class My_MVEM(pp.MVEM):

    def __init__(self, keyword):
        self.file_name = None
        super(My_MVEM, self).__init__(keyword)


    def discretize(self, g, data):
        self.file_name = "grid" + str(g.dim) + ".csv"
        open(self.file_name, "w").close()
        super(My_MVEM, self).discretize(g, data)

    def massHdiv(self, K, c_center, c_volume, f_centers, normals, sign, diam, weight=0):
        A, Pi_s = pp.MVEM.massHdiv(K, c_center, c_volume, f_centers, normals, sign, diam, weight)
        # do it only in the construction of the linear system
        if weight != 0:
            # with weight = 0 (the last parameter) in A is not present the stabilization term
            A_w0, _ = pp.MVEM.massHdiv(K, c_center, c_volume, f_centers, normals, sign, diam, 0)
            S = A - A_w0

            norm_A = np.linalg.norm(A)
            norm_S = np.linalg.norm(S)

            with open(self.file_name, "a") as f:
                f.write(str(norm_A) + ", " + str(norm_S) + ", " + str(norm_S/norm_A) + "\n")
        return A, Pi_s
