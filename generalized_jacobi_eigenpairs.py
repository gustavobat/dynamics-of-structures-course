import numpy as np


def assert_matrices_dimensions(mat_a, mat_b):
    assert mat_a.shape[0] == mat_a.shape[1], "Matrix is not square."
    assert mat_b.shape[0] == mat_b.shape[1], "Matrix is not square."
    assert mat_a.shape[0] == mat_b.shape[0], "Matrices are not the same size."


def sign(val):
    if val >= 0:
        return 1
    else:
        return -1


def sort_eigenpairs(eigen_vals, eigen_vecs):
    ndof = eigen_vals.shape[0]
    for i in range(ndof):
        min_pos = i
        for j in range(i + 1, ndof):
            if eigen_vals[j] < eigen_vals[min_pos]:
                min_pos = j

        eigen_vals[i], eigen_vals[min_pos] = eigen_vals[min_pos], eigen_vals[i].copy()
        eigen_vecs[:, i], eigen_vecs[:, min_pos] = eigen_vecs[:, min_pos], eigen_vecs[:, i].copy()
    return eigen_vals, eigen_vecs


# TODO comment code and test number of rotation/sweeps against the book example
def calculate_eigenpairs(m, k, tol):
    m = np.array(m)
    k = np.array(k)
    assert_matrices_dimensions(m, k)
    ndof = m.shape[0]

    # Initialize eigenpairs
    eigen_vals = np.empty([ndof, 1])
    for dof in range(ndof):
        eigen_vals[dof, 0] = k[dof, dof] / m[dof, dof]
    eigen_vecs = np.identity(ndof)

    sweep_counter = 0
    rot_counter = 0

    has_converged = False
    while not has_converged:
        sweep_counter += 1
        rot_tol = np.power(10., -2 * sweep_counter)

        for i in range(ndof - 1):
            for j in range(i + 1, ndof):

                needs_rotation = False
                if np.sqrt(k[i, j] * k[i, j] / (k[i, i] * k[j, j])) > rot_tol:
                    needs_rotation = True
                if np.sqrt(k[i, j] * k[i, j] / (k[i, i] * k[j, j])) > rot_tol:
                    needs_rotation = True

                if needs_rotation:
                    rot_counter += 1
                    a0 = k[i, i] * m[i, j] - m[i, i] * k[i, j]
                    a1 = k[j, j] * m[i, j] - m[j, j] * k[i, j]
                    a2 = k[i, i] * m[j, j] - m[i, i] * k[j, j]
                    gamma = a2 / 2 + sign(a2) * np.sqrt((a2 / 2) * (a2 / 2) + a0 * a1)

                    beta = 0
                    if gamma == 0:
                        alpha = -k[i, j] / k[j, j]
                    else:
                        alpha = -a0 / gamma
                        beta = a1 / gamma

                    p_k = np.identity(ndof)
                    p_k[i, j] = beta
                    p_k[j, i] = alpha
                    m = p_k.transpose().dot(m).dot(p_k)
                    k = p_k.transpose().dot(k).dot(p_k)
                    eigen_vecs = eigen_vecs.dot(p_k)

                # Test convergence
                has_converged = True
                new_eigen_vals = np.empty([ndof, 1])
                for dof in range(ndof):
                    new_eigen_vals[dof, 0] = k[dof, dof] / m[dof, dof]
                    diff = new_eigen_vals[dof, 0] - eigen_vals[dof, 0]
                    diff /= new_eigen_vals[dof, 0]
                    if diff > tol:
                        has_converged = False
                eigen_vals = new_eigen_vals

                # Verify degree of coupling
                if np.sqrt(m[i, j] * m[i, j] / m[i, i] * m[j, j]) > tol:
                    has_converged = False
                if np.sqrt(k[i, j] * k[i, j] / k[i, i] * k[j, j]) > tol:
                    has_converged = False

    normalization_matrix = np.identity(ndof)
    for dof in range(ndof):
        normalization_matrix[dof, dof] /= np.sqrt(m[dof, dof])
    eigen_vecs = eigen_vecs.dot(normalization_matrix)

    eigen_vals, eigen_vecs = sort_eigenpairs(eigen_vals, eigen_vecs)

    return eigen_vals, eigen_vecs
