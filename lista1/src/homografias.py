# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os
from sklearn.metrics import mean_squared_error

def read_points_from_file(file):
    points = []

    with open(file) as f:
        for line in f:
            points.append([int(np.float(x)) for x in line.strip().split()])

    #retifica posicao do ponto, pois no arquivo esta no formato y, x
    for p in points:
        aux = p[0]
        p[0] = p[1]
        p[1] = aux

    return points

def mark_points_on_image(points, image):
    aux_image = image.copy()

    for p in points:
        cv2.circle(aux_image, (p[0], p[1]), 3, (0, 255, 0), -1)

    cv2.imshow("original", aux_image)

    cv2.waitKey(0)
    return 0

#retorna tupla de 3 dimenções das cores do pixel interpolado
def interpolacao_bilinear(posicao, imagem):
    x=posicao[1]
    y=posicao[0]
    x1=int(np.floor(posicao[1]))
    x2=int(np.ceil(posicao[1]))
    y1=int(np.floor(posicao[0]))
    y2=int(np.ceil(posicao[0]))
    q11=imagem[y1, x1]
    q21=imagem[y1, x2]
    q12=imagem[y2, x1]
    q22=imagem[y2, x2]

    return (q11 * (x2 - x) * (y2 - y) +
            q21 * (x - x1) * (y2 - y) +
            q12 * (x2 - x) * (y - y1) +
            q22 * (x - x1) * (y - y1)
           ) / ((x2 - x1) * (y2 - y1) + 0.0)

def normalization(points):

    points = np.asarray(points)

    m = np.mean(points, 0)
    s = np.std(points)

    T = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])
    T = np.linalg.inv(T)
    normalized_points = np.dot( T, np.concatenate( (points.T, np.ones((1, points.shape[0]))) ) )
    normalized_points = normalized_points[0:2,:].T

    return T, normalized_points

def create_homographi_dlt(points_1, points_2):

    #aplica normalizacao dos dados para se obter melhores resultados
    T1, points_1_norm = normalization(points_1)
    T2, points_2_norm = normalization(points_2)

    A = []

    for i in range(len(points_1)):
        x,y = points_1_norm[i,0], points_1_norm[i,1]
        u,v = points_2_norm[i,0], points_2_norm[i,1]
        A.append( [x, y, 1, 0, 0, 0, -u*x, -u*y, -u] )
        A.append( [0, 0, 0, x, y, 1, -v*x, -v*y, -v] )

    #aplica svd para encontrar os 8 parametros
    U, S, Vh = np.linalg.svd(A)

    #recupera parametros do svd
    L = Vh[-1,:] / Vh[-1,-1]

    #matriz de projecao
    H = L.reshape(3,3)

    #denormalizacao
    H = np.dot( np.dot( np.linalg.pinv(T2), H ), T1 );
    H = H / H[-1,-1]

    return H


def aplica_matriz_transformacao(image, H):
    #aux_image = image.copy()

    aux_image = np.zeros((2*image.shape[0], 2*image.shape[1], 3), dtype=np.uint8)

    for y in range(image.shape[0]): #percorre linhas da imagem original
        for x in range(image.shape[1]): #percorre colunas da imagem original
            posicao_final=np.dot(H, [x, y, 1])
            posicao_final = posicao_final/posicao_final[2]
            if posicao_final[0]>=0 and posicao_final[0]<aux_image.shape[1] and posicao_final[1]>=0 and posicao_final[1]<aux_image.shape[0]:
                aux_image[int(posicao_final[1]), int(posicao_final[0])] = image[y,x]

    return aux_image


if __name__ == '__main__':

    #subitem 1
    homografia_point_file_1 = "../dados/homografia_1.txt"
    homografia_image_file_1 = "../dados/img_homografia_1.png"

    points_1 = read_points_from_file(homografia_point_file_1)

    original_image_1 = cv2.imread(homografia_image_file_1, cv2.IMREAD_COLOR)

    mark_points_on_image(points_1, original_image_1)


    homografia_point_file_2 = "../dados/homografia_2.txt"
    homografia_image_file_2 = "../dados/img_homografia_2.png"

    points_2 = read_points_from_file(homografia_point_file_2)

    original_image_2 = cv2.imread(homografia_image_file_2, cv2.IMREAD_COLOR)

    mark_points_on_image(points_2, original_image_2)


    #subitem 2

    h12 = create_homographi_dlt(points_1, points_2)
    h21 = create_homographi_dlt(points_2, points_1)

    print(h12)

    print(h21)


    #subitem 3

    #constroi imagem 1 com homografia e plota
    transformada_imagem_1 = aplica_matriz_transformacao(original_image_1, h12)

    cv2.imshow("trasformacao img 1", transformada_imagem_1)

    cv2.waitKey(0)


    #constroi imagem 2 com homografia e plota
    transformada_imagem_2 = aplica_matriz_transformacao(original_image_2, h21)

    cv2.imshow("transformacao img 2", transformada_imagem_2)

    cv2.waitKey(0)
