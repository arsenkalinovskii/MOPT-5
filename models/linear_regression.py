from polynomial_regression import PolynomialRegression


class LinearRegression(PolynomialRegression):
    def __init__(self):
        super().__init__(degree=1)

    def __repr__(self):
        return "LinearRegression()"
