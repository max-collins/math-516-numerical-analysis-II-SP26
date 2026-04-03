import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.basis.affine import AffinePolynomialBasis
from numanalysislib.basis.piecewise import PiecewisePolynomial
from numanalysislib.basis.power import PowerBasis
from numanalysislib.plotting import Plotter


class TestPiecewise():
    @pytest.fixture
    def pw_poly(self):
        basis = PowerBasis(degree=1)
        h = 0.5
        mesh = [(k*h, (k+1)*h) for k in range(4)]
        return PiecewisePolynomial(basis, mesh)
    
    def test_fit_evaluate_exact(self, pw_poly):
        """
        expected coeffs:
        [0, 1], [1, -1], [-2, 2], [0.25, 0.5]
        """
        y_nodes = [np.array([0.0, 0.5]), np.array([0.5, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.25])]
        h = 0.5
        x_vals = np.array([k*h for k in range(5)])
        y_vals = np.array([0.0, 0.5, 0.0, 1.0, 1.25])
        coeffs = pw_poly.fit(y_nodes)
        pred_y_vals = pw_poly.evaluate(coeffs, x_vals)
        np.testing.assert_allclose(pred_y_vals, y_vals)
        
    
    def test_fit_failure(self, pw_poly):
        with pytest.raises(ValueError):
            pw_poly.fit([np.array([0.0, 0.5]), np.array([0.6, 0.0]), np.array([0.0, 1.0]), np.array([1.0, 1.25])])


    def test_init_error(self):
        basis = PowerBasis(degree=1)
        mesh = [(0.0, 0.2), (0.1, 0.3)]
        with pytest.raises(ValueError):
            return PiecewisePolynomial(basis, mesh)


    def test_plotter(self, pw_poly):
        pass
