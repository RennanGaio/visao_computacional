# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import random


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



def normalization_2d(points):

    points = np.asarray(points)

    m = np.mean(points, 0)
    s = np.std(points)

    T = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    T = np.linalg.inv(T)
    normalized_points = np.dot( T, points.T )
    normalized_points = normalized_points[0:2,:].T

    return T, normalized_points


def normalization_3d(points):

    points = np.asarray(points)

    m = np.mean(points, 0)
    s = np.std(points)

    T = np.array([[s, 0, 0, m[0]], [0, s, 0, m[1]], [0, 0, s, m[2]], [0, 0, 0, 1]])

    T = np.linalg.inv(T)
    normalized_points = np.dot( T, points.T )
    normalized_points = normalized_points[0:3,:].T

    return T, normalized_points

def create_projection_dlt(points_2d, points_3d):

    #aplica normalizacao dos dados para se obter melhores resultados
    T1, points_2d_norm = normalization_2d(points_2d)
    T2, points_3d_norm = normalization_3d(points_3d)

    A = []

    for i in range(len(points_2d_norm)):
        x,y,z = points_3d_norm[i,0], points_3d_norm[i,1], points_3d_norm[i,2]
        u,v = points_2d_norm[i,0], points_2d_norm[i,1]
        A.append( [x, y, z, 1, 0, 0, 0, 0, -u*x, -u*y, -u*z, -u] )
        A.append( [0, 0, 0, 0, x, y, z, 1, -v*x, -v*y, -v*z, -v] )

    #aplica svd para encontrar os 11 parametros
    U, S, Vh = np.linalg.svd(A)

    #recupera parametros do svd
    L = Vh[-1,:] / Vh[-1,-1]

    #matriz de projecao
    H = L.reshape(3,4)


    #denormalizacao
    H = np.dot( np.dot( np.linalg.pinv(T1), H ), T2 );
    H = H / H[-1,-1]

    #Erro medio da DLT
    new_points_2d = np.dot( H, points_3d.T )
    new_points_2d = new_points_2d/new_points_2d[2,:]

    #distancia media
    error = np.sqrt( np.mean(np.sum( (new_points_2d.T - points_2d)**2,1 )) )

    return H, error

def calculate_max_value(points):
    max = 0

    for p in points:
        if p[0]> max:
            max = p[0]
        if p[1]> max:
            max = p[0]

    return max


def adiciona_ruido_ro(points, ro):
    points_ruidosos = []

    for p in points:
        r = random.randint(1,2)
        point= []
        if r == 2:
            for axis in p:
                point.append(axis-ro)
        else:
            for axis in p:
                point.append(axis+ro)

        point = np.array(point)
        point = point/point[-1]

        points_ruidosos.append(point)

    return np.array(points_ruidosos)


def adiciona_ruido_percentagem(points_3d, percentage, max3d):
    points_ruidosos = []

    number_of_samples = int(len(points_3d*percentage))

    for i in range(number_of_samples):
        r = random.randint(0, len(points_3d)-1)

        point= []

        for axis in points_3d[r]:
                point.append(random.randint(int(-max3d/2), int(max3d/2)))

        point = np.array(point)
        point = point/point[-1]

        points_ruidosos.append(point)

    return np.array(points_ruidosos)


if __name__ == '__main__':

    arquivo_de_pontos_2d = "../dados/Pontos2D_1.npy"
    arquivo_de_pontos_3d = "../dados/Pontos3D_1.npy"

    points_2d = load_numpy_data(arquivo_de_pontos_2d )
    points_3d = load_numpy_data(arquivo_de_pontos_3d )

    #subitem 1
    plot_3d_data(points_3d)

    plot_2d_data(points_2d)

    #subitem 2
    projection_matrix, projectionError = create_projection_dlt(points_2d, points_3d)

    print("Matriz de projecao para os pontos sem ruido de 3d para 2d")

    print(projection_matrix)

    #subitem 3

    print("erro de projecao:")
    print(projectionError)

    #subitem 4

    print("adicionando ruido ro")

    max3d = calculate_max_value(points_3d)

    ro = 0.05*max3d

    pontos_ruidosos_3d = adiciona_ruido_ro(points_3d, ro)

    projection_matrix_ruido_ro, projectionError = create_projection_dlt(points_2d, pontos_ruidosos_3d)

    print("Matriz de projecao para os pontos com ruido ro de 3d para 2d")

    print(projection_matrix_ruido_ro)

    print("erro de projecao:")
    print(projectionError)


    #subitem 5

    print("adicionando 20 porcento de ruido nos pontos 3d")

    pontos_ruidosos_3d = adiciona_ruido_percentagem(points_3d, 0.2, max3d)

    projection_matrix_ruido_percentagem, projectionError = create_projection_dlt(points_2d, pontos_ruidosos_3d)


    print("Matriz de projecao para os pontos com 20 porcento de ruido de 3d para 2d")

    print(projection_matrix_ruido_percentagem)

    print("erro de projecao:")
    print(projectionError)
