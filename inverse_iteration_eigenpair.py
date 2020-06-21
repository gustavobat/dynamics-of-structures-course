import numpy as np


def assert_matrices_dimensions(mat_a, mat_b):
    assert mat_a.shape[0] == mat_a.shape[1], "Matrix is not square."
    assert mat_b.shape[0] == mat_b.shape[1], "Matrix is not square."
    assert mat_a.shape[0] == mat_b.shape[0], "Matrices are not the same size."


def calculate_eigenpair(m, k, tol, shift=0.):
    m = np.array(m)
    k = np.array(k)
    assert_matrices_dimensions(m, k)
    ndof = m.shape[0]

    # Initialize variables
    x = np.ones([ndof, 1])
    y = m.dot(x)
    eigen_val = 0
    eigen_vec = np.empty([ndof, 1])
    iteration_counter = 0

    has_converged = False
    while not has_converged:
        iteration_counter += 1
        shifted_k = k - shift * m
        x = np.linalg.inv(shifted_k).dot(y)
        new_y = m.dot(x)
        # Calculation of Rayleigh quotient
        rayleigh_coef = np.transpose(x).dot(y) / np.transpose(x).dot(new_y)
        eigen_val = rayleigh_coef + shift
        # Normalization of the iteration vector
        new_y = (1 / np.sqrt((np.transpose(x).dot(new_y)))) * new_y
        # Convergence check
        if np.abs((eigen_val - shift) / eigen_val) < tol:
            has_converged = True

        # Update values
        y = new_y
        shift = eigen_val

        if has_converged:
            eigen_vec = np.linalg.inv(m).dot(y)

    return eigen_val, eigen_vec


def main():
    m = [[20000, 0, 0, 0],
         [0, 20000, 0, 0],
         [0, 0, 20000, 0],
         [0, 0, 0, 20000]]

    k = [[2.53352e7, -1.43816e7, 3.34428e6, -4.27270e5],
         [-1.43816e7, 2.20846e7, -1.36664e7, 2.66592e6],
         [3.34428e6, -1.36664e7, 2.06543e7, -9.82118e6],
         [-4.27270e5, 2.66592e6, -9.82118e6, 7.51725e6]]

    tol = 1e-6
    shift = 280.
    val, vec = calculate_eigenpair(m, k, tol, shift)
    print(val)
    print(vec)


if __name__ == "__main__":
    main()
