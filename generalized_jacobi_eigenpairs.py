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


def calculate_eigenpairs(m, k, tol):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    assert_matrices_dimensions(m, k)
    ndof = m.shape[0]

    # Initialize eigenpairs
    eigen_values = np.empty([ndof, 1])
    for i in range(ndof):
        eigen_values[i] = k[i, i] / m[i, i]
    eigen_vectors = np.identity(ndof)

    print("EigenValues:\n", eigen_values)
    print("EigenVectors:\n", eigen_vectors)
    sweep = 0
    rot = 1

    calculated_tol = 1e12
    #while calculated_tol > tol:
    for i in range(ndof - 1):
        for j in range(i + 1, ndof):

            needs_rotation = False
            if np.sqrt(k[i, j] * k[i, j] / (k[i, i] * k[j, j])) > tol:
                needs_rotation = True
            if np.sqrt(k[i, j] * k[i, j] / (k[i, i] * k[j, j])) > tol:
                needs_rotation = True

            if needs_rotation:
                a0 = k[i, i] * m[i, j] - m[i, i] * k[i, j]
                a1 = k[j, j] * m[i, j] - m[j, j] * k[i, j]
                a2 = k[i, i] * m[j, j] - m[i, i] * k[j, j]
                gamma = a2 / 2 + sign(a2) * np.sqrt((a2 / 2) * (a2 / 2) + a0 * a1)

                alpha = 0
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
                eigen_vectors.dot(p_k)
                print("i =", i + 1, " j =", j + 1, " alpha = ", alpha, " beta = ", beta)
                print("K = \n", k)
                print("M = \n", m)
                print("Phi = \n", eigen_vectors)



def main():
    m = np.array([[4, 1, 0, 0],
                  [1, 4, 1, 0],
                  [0, 1, 4, 1],
                  [0, 0, 1, 2]])

    k = np.array([[2, -1, 0, 0],
                  [-1, 2, -1, 0],
                  [0, -1, 2, -1],
                  [0, 0, -1, 1]])
    tol = 1e-6
    calculate_eigenpairs(m, k, tol)


if __name__ == "__main__":
    main()
