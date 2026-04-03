from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.basis.affine import AffinePolynomialBasis
from numanalysislib.basis.power import PowerBasis #DELETE THIS
import numpy as np
import warnings
import matplotlib.pyplot as plt


class PiecewisePolynomial:
    def __init__(self, basis_type: PolynomialBasis, mesh: list):
        self.basis_type = basis_type

        #order mesh
        mesh.sort()
        self.mesh = mesh
        
        #check that mesh has no overlaps
        for index in range(len(self.mesh)-1):
            element1 = self.mesh[index]
            element2 = self.mesh[index + 1]

            if element1[1] != element2[0]:
                raise ValueError("mesh must be have connected elements")

        self.bases = {}
        for element in self.mesh:
            a = element[0]
            b = element[1]
            self.bases[element] = AffinePolynomialBasis(basis_type, a, b)

    
    def fit(self, y_mesh: list):
        # y_mesh is list of arrays with enough pts per element for required degree
        # dofs considered from left endpoint e.g. if intervals are (0.0, 0.5) and (0.5, 0.1) with a degree 1
        # polynomial basis then the y_mesh could be [np.array([1]), np.array([2]), np.array([1])]

        bases_coeffs = {}

        last_right_endpoint = y_mesh[0][0] #initializer tracker for last encountered right endpoint

        for index, y_element in enumerate(y_mesh):
            # check continuity
            if y_element[0] != last_right_endpoint:
                raise ValueError(f"y_mesh must induce continuous piecewise polynomial. Ensure endpoints are consistent")
            last_right_endpoint = y_element[-1]

            # extract info from element
            element = self.mesh[index]
            a = element[0]
            b = element[1]
            n_dofs = self.bases[element].n_dofs

            # create equidistant x_nodes to fit with
            x_nodes = []
            for i in range(n_dofs):
                x_nodes.append((1-i/(n_dofs-1))*a + i/(n_dofs-1)*b)
            x_nodes = np.asarray(x_nodes)
            #fit 
            bases_coeffs[element] = self.bases[element].fit(x_nodes, y_element)
        

        return bases_coeffs

    def float_evaluate(self, coefficients: dict, x: float):

        min_index = 0
        max_index = len(self.mesh)
        current_index = min_index + (max_index - min_index)//2
        for _ in range(len(self.mesh)):
            element = self.mesh[current_index]
            a = element[0]
            b = element[1]
            if x < a:
                max_index = current_index
            elif x > b:
                min_index = current_index
            else:
                return self.bases[element].evaluate(coefficients[element], x) 
            
            current_index = min_index + (max_index - min_index)//2
        # throw a warning if the loop finishes
        warnings.warn("no interval found")
    
    def evaluate(self, coefficients: dict, x: np.array):
        return np.vectorize(lambda y: self.float_evaluate(coefficients, y))(x)

if __name__ == "__main__":
    power = PowerBasis(1)
    h=0.5
    pw_poly = PiecewisePolynomial(power, [(k*h, (k+1)*h) for k in range(4)])
    coeffs = pw_poly.fit([np.array([0.0, 0.5]), np.array([0.5, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.25])])
    print(coeffs)
    left_endpts = [pw_poly.mesh[i][0] for i in range(len(pw_poly.mesh))]
    print("mesh", pw_poly.mesh)
    print(pw_poly.evaluate(coeffs, 1.6))
    plt.plot(pw_poly.evaluate(coeffs, np.linspace(0,2, 100)))
    plt.show()
