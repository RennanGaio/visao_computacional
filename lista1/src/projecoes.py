# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os
from sklearn.metrics import mean_squared_error


def load_numpy_data(file):
    data = np.load(file, allow_pickle=True)
    #print(data)
    #print(len(data))

    return data

def plot_3d_data(points_3d):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for p in points_3d:
        ax.scatter(p[0], p[1], p[2], marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()

def plot_2d_data(points_2d):
    fig = plt.figure()
    ax = fig.add_subplot()

    for p in points_2d:
        ax.scatter(p[0]/p[2], p[1]/p[2], marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()


def projection_matrix_construction(points_2d, points_3d):
    projection_matrix = []

    '''
    #svd decomposition
    U, s, Vh = np.linalg.svd(matrix)

    #recover matrix
    matrix = np.dot(U[:, :s.shape[0]] * s, Vh)
    '''

    return projection_matrix

def projection_error(points_2d, projected_points):
    #erro por minimos quadrados
    error = mean_squared_error(points_2d, projected_points)

    return error


if __name__ == '__main__':

    arquivo_de_pontos_2d = "../dados/Pontos2D_1.npy"
    arquivo_de_pontos_3d = "../dados/Pontos3D_1.npy"

    points_2d = load_numpy_data(arquivo_de_pontos_2d )
    points_3d = load_numpy_data(arquivo_de_pontos_3d )

    #subitem 1
    plot_3d_data(points_3d)

    plot_2d_data(points_2d)

    #subitem 2
    projection_matrix = projection_matrix_construction(points_2d, points_3d)

    print(projection_matrix)

    #subitem 3

    projected_points = np.dot(points_3d, np.transpose(projection_matrix))

    projectionError = projection_error(points_2d, projected_points)
    print("erro de projecao:")
    print(projectionError)

    plot_2d_data(projected_points)
