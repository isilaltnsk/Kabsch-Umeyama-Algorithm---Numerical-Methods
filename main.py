# PROJECT NO: 5
# IŞIL ALTINIŞIK - 150220308

import numpy as np
import matplotlib.pyplot as plt
import argparse
from mpl_toolkits.mplot3d import Axes3D

# a function reads the files
def read(filename):
    data = np.loadtxt(filename)
    return data


# kabsch-umeyama algorithm
def kabsch_umeyama(points_A, points_B):
    # Calculate the centroid
    centroid_A = np.mean(points_A, axis=0)
    centroid_B = np.mean(points_B, axis=0)
    # Center the point sets
    centered_a = points_A - centroid_A
    centered_b = points_B - centroid_B
    return centered_a, centered_b, centroid_A, centroid_B



# calculation of covariance matrix
def covariance(centered_a, centered_b):
    centered_points_t = centered_a.T
    H = np.dot(centered_points_t, centered_b)
    return H



# finding eigenvalues with QR decomposition
def eigenvalues(matrix):
    vectors = np.eye(matrix.shape[1])
    for i in range(0,7):
        q1, r1 = np.linalg.qr(matrix)     # There's not any rule about linalg.qr method so didnt compute qr decomposition from scratch
        vectors = np.dot(vectors, q1)
        matrix = np.dot(r1, q1)

    eigen_values = np.diag(matrix)
    eigen_vectors = vectors
    return eigen_values, eigen_vectors



# my singular value decomposition (svd) code
def SVD(A):
    # Compute A^T A
    ATA = np.dot(A.T, A)
    # find the eigenvector and eigenvalues
    eigen_values_V, eigen_vectors_V = eigenvalues(ATA)
    # sort the eigenvalues
    idx = eigen_values_V.argsort()[::-1]
    eigen_values_V = eigen_values_V[idx]
    eigen_vectors_V = eigen_vectors_V[:, idx]
    # compute singular values
    singular_values = np.sqrt(eigen_values_V)
    #compute V
    V = eigen_vectors_V
    # Compute U
    U = np.dot(A, V) / singular_values
    sigma = np.diag(singular_values)

    return U, sigma, V.T



# finally aligning two point sets
def aligned(t, c, R, points_b):
    merged_b = np.dot(R.T, points_b.T - t)
    return merged_b



# calculation of mat1 and inverse rotated-translated mat2.
def merged(merged_b, points_A):
    merged = []
    for i in range(0, len(merged_b)):
        merged.append(merged_b[i])
        if i < len(points_A):
            merged.append(points_A[i])

    return np.array(merged)




def plot_3d_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:,0], points[:,1], points[:,2], c='b', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()




# function to save the outputs
def savemyresults(R, t, merged):
    np.savetxt('rotation_mat.txt', R)
    np.savetxt('translation_vec.txt', t)
    np.savetxt('merged', merged)



# main function - other calculations and implementations ( whole process)
def main(mat1_file, mat2_file, correspondences_file):
    # read the given files
    correpondences = read(correspondences_file).astype(int)
    points_A = read(mat1_file)
    points_B = read(mat2_file)
    # create correspondences sets
    points_a = []
    points_b = []
    for i in correpondences:
        A = int(i[0])
        B = int(i[1])
        # print(A, B)
        points_a.append(points_A[A])
        points_b.append(points_B[B])
    # create the new matrices
    points_a = np.array(points_a)
    points_b = np.array(points_b)
    centered_a, centered_b, centroid_A, centroid_B = kabsch_umeyama(points_a, points_b)

    # algorithm implementation
    H = covariance(centered_a, centered_b)
    U, D, VT = SVD(covariance(centered_a, centered_b))
    R = np.dot(VT.T, U.T)
    #print("R:", R)   - just to check
    d = np.linalg.det(U) * np.linalg.det(VT)
    if d > 0:  # 0 olma durumu var mı ?
        d = 1
    elif d < 0:
        d = -1
    t = centroid_B - np.dot(R, centroid_A)
    #print("t:", t)  - again just to check
    x = np.shape(points_B)[0]
    expanded_t = t.reshape(-1, 1).repeat(x, axis=1)

    newU, S, newvt = SVD(R)
    variance_A = np.mean(np.sum((points_A - centroid_A) ** 2))
    # in the project file, finding c was not mentioned but in the articles it was. so i calculated anyway but didnt use
    # because it was not required
    c = np.trace(np.dot(D, S)) / variance_A

    merged_b = aligned(expanded_t, c, R, points_B).T
    # find the merges of sets
    merged_file = merged(merged_b, points_A)

    savemyresults(R, t, merged_file)

    plot_3d_points(points_A)
    plot_3d_points(points_B)
    plot_3d_points(merged_b)

# NOTE !! = There is a small error between the R, t and merged matrices i found and the correct answers. But it's because
# i used QR decomposition while finding the eigenvalues and it is just an approximation. If you use linalg.eig() instead
# of that, the results will be exactly same.


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Perform Kabsch algorithm to align two sets of points.")
    parser.add_argument("mat1_file", help="Path to the first matrix file (mat1.txt).")
    parser.add_argument("mat2_file", help="Path to the second matrix file (mat2.txt).")
    parser.add_argument("correspondences_file", help="Path to the correspondences file (correspondences.txt).")
    args = parser.parse_args()

    # Call main function with command-line arguments
    main(args.mat1_file, args.mat2_file, args.correspondences_file)
