import numpy as np
import matplotlib.pyplot as plt


class EigenvalueSolver:
    def __init__(self, A, tol=1e-9, maxit=100):
        self.A = A
        self.tol = tol
        self.maxit = maxit
        self.n = A.shape[0]

    def power_method(self):
        y = np.ones(self.n)
        mu = np.zeros(self.maxit)
        for i in range(self.maxit):
            y = self.A @ y
            y /= np.linalg.norm(y)
            mu[i] = y.T @ self.A @ y
            if i > 0 and np.abs(mu[i] - mu[i - 1]) < self.tol:
                break
        return mu[: i + 1]

    def inverse_power_method(self):
        y = np.ones(self.n)
        nu = np.zeros(self.maxit)
        for i in range(self.maxit):
            y = np.linalg.solve(self.A, y)
            y /= np.linalg.norm(y)
            nu[i] = y.T @ self.A @ y
            if i > 0 and np.abs(nu[i] - nu[i - 1]) < self.tol:
                break
        return nu[: i + 1]

    def shift_and_invert_method(self, shift):
        B = self.A - shift * np.eye(self.n)
        y = np.ones(self.n)
        sigma = np.zeros(self.maxit)
        for i in range(self.maxit):
            y = np.linalg.solve(B, y)
            y /= np.linalg.norm(y)
            sigma[i] = y.T @ self.A @ y
            if i > 0 and np.abs(sigma[i] - sigma[i - 1]) < self.tol:
                break
        return sigma[: i + 1]

    def plot_convergence(self, shift):
        mu = self.power_method()
        nu = self.inverse_power_method()
        sigma = self.shift_and_invert_method(shift)

        plt.semilogy(range(len(mu)), np.abs(mu - mu[-1]), "k^-", label="Power Method")
        plt.semilogy(
            range(len(nu)), np.abs(nu - nu[-1]), "bv-", label="Inverse Power Method"
        )
        plt.semilogy(
            range(len(sigma)),
            np.abs(sigma - sigma[-1]),
            "r*-",
            label="Shift and Invert Method",
        )
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error")
        plt.legend()
        plt.title("Convergence of Eigenvalue Approximations")
        plt.show()


# Usage
if __name__ == "__main__":
    A = np.array([[4, 1, 0, 0], [1, 3, 1, 0], [0, 1, 2, 1], [0, 0, 1, 1]], dtype=float)
    solver = EigenvalueSolver(A, tol=1e-9, maxit=100)
    shift_value = 3
    solver.plot_convergence(shift_value)
