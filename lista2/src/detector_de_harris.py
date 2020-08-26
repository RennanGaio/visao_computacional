# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os
from sklearn.metrics import mean_squared_error


def CalculaCovarianceMatrix(sobelx, sobely, x, y, Wc):
    c11=0
    c12=0
    c22=0
    for i in range(y-int(Wc/2), y+math.ceil(Wc/2)):
        for j in range(x-int(Wc/2), x+math.ceil(Wc/2)):
            c11+= sobelx[i][j]**2
            c12+=sobelx[i][j]*sobely[i][j]
            c22+=sobely[i][j]**2

    C = np.array([[c11, c12], [c12, c22]])
    return C


def HarrisDetector(imagem, Wd, Wc, k=0.04):
    R = np.zeros((imagem.shape[0], imagem.shape[1]))

    #calcula as derivadas de sobel
    sobelx = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=Wd)
    sobely = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=Wd)

    for y in range(int(Wc/2), imagem.shape[0]-int(Wc/2)): #percorre linhas da imagem original
        for x in range(int(Wc/2), imagem.shape[1]-int(Wc/2)): #percorre colunas da imagem original
            C = CalculaCovarianceMatrix(sobelx, sobely, x, y, Wc)
            traco = np.trace(C)
            det = np.linalg.det(C)

            R[y][x] = det - k*(traco**2)

    return R



def AgrupaHarris(dst, pontos_marcados, Wh):
    pontos_retorno = []
    #separa a imagem em regioes
    #para cada regiao, utiliza apenas 1 ponto

    passo_y = math.ceil(dst.shape[0]/float(Wh))
    passo_x = math.ceil(dst.shape[1]/float(Wh))


    for y in range(passo_y): #percorre linhas da imagem original
        for x in range(passo_x): #percorre colunas da imagem original

            maior_ponto_no_quadrante = [0, [0, 0]]
            for ponto in pontos_marcados:
                #verifica se ponto esta no grid
                if ponto[0] > y*Wh and ponto[0] < (y+1)*Wh and ponto[1] > x*Wh and ponto[1] < (x+1)*Wh:
                    #salva o maior ponto no quadrante
                    if dst[ponto[0]][ponto[1]] > maior_ponto_no_quadrante[0]:
                        maior_ponto_no_quadrante[0] = dst[ponto[0]][ponto[1]]
                        maior_ponto_no_quadrante[1] = ponto

            if maior_ponto_no_quadrante[0] > 0:
                pontos_retorno.append(maior_ponto_no_quadrante[1])


    return pontos_retorno


if __name__ == '__main__':

    #inicializacao de parametros
    Wd=3
    # Wc = 2
    Wc=5*Wd
    k=0.04
    Wh = 40
    printe = True
    TR = 0.02 # valor é o multiplicador do maior valor de dst

    imagem_utilizada = "goi"
    pasta_resultados = "../results/harris/"+imagem_utilizada+"/"+str(Wh)+"/"
    # pasta_resultados = "../results/harris/"+imagem_utilizada+"/wc2/"

    #imagem 1
    imagem_goi_file_1 = "../dados/"+imagem_utilizada+"1.jpg"
    imagem_goi_file_2 = "../dados/"+imagem_utilizada+"2.jpg"


    #imagem 2
    #imagem_building_file_1 = "../dados/building1.jpg"
    #imagem_building_file_2 = "../dados/building2.jpg"



    original_goi_1 = cv2.imread(imagem_goi_file_1, cv2.IMREAD_COLOR)
    gray_goi_1 = cv2.cvtColor(original_goi_1, cv2.COLOR_BGR2GRAY)


    original_goi_2 = cv2.imread(imagem_goi_file_2, cv2.IMREAD_COLOR)
    gray_goi_2 = cv2.cvtColor(original_goi_2, cv2.COLOR_BGR2GRAY)

    if printe:
        cv2.imshow('original '+imagem_utilizada+' 1', original_goi_1)
        cv2.imshow('original '+imagem_utilizada+' 2', original_goi_2)
        cv2.waitKey(0)

    cv2.imwrite(pasta_resultados+'original '+imagem_utilizada+' 1.jpg', original_goi_1)
    cv2.imwrite(pasta_resultados+'original '+imagem_utilizada+' 2.jpg', original_goi_2)


    #############open cv implementation

    gray_1 = np.float32(gray_goi_1)
    dst_1 = cv2.cornerHarris(gray_1, Wc, Wd, k)

    gray_2 = np.float32(gray_goi_2)
    dst_2 = cv2.cornerHarris(gray_2, Wc, Wd, k)


    ################## my implementation

    # dst = HarrisDetector(gray_goi_1, Wd, Wc, k)

    #print(dst)

    #Para Tr foi escolhido o valor 0.03 do valor maximo
    marked_image_1 = original_goi_1.copy()
    marked_image_2 = original_goi_2.copy()


    marked_image_1[dst_1>TR*dst_1.max()] = [0, 0, 255]
    marked_image_2[dst_2>TR*dst_2.max()] = [0, 0, 255]

    #salva os pontos marcados
    pontos_marcados_1 = []
    pontos_marcados_2 = []

    RED = np.array([0, 0, 255])

    for y in range(marked_image_1.shape[0]):
        for x in range(marked_image_1.shape[1]):
            #print(marked_image[y][x])
            if marked_image_1[y][x].all() == RED.all():
                pontos_marcados_1.append([y, x])

    for y in range(marked_image_2.shape[0]):
        for x in range(marked_image_2.shape[1]):
            #print(marked_image[y][x])
            if marked_image_2[y][x].all() == RED.all():
                pontos_marcados_2.append([y, x])

    # print(pontos_marcados)

    if printe:
        cv2.imshow('dst com grupos '+imagem_utilizada+' 1', marked_image_1)
        cv2.imshow('dst com grupos '+imagem_utilizada+' 2', marked_image_2)
        cv2.waitKey(0)

    cv2.imwrite(pasta_resultados+'dst com grupos '+imagem_utilizada+' 1.jpg', marked_image_1)
    cv2.imwrite(pasta_resultados+'dst com grupos '+imagem_utilizada+' 2.jpg', marked_image_2)



    pontos_marcados_1 = AgrupaHarris(dst_1, pontos_marcados_1, Wh)
    pontos_marcados_2 = AgrupaHarris(dst_2, pontos_marcados_2, Wh)

    #print(pontos_marcados)

    for ponto in pontos_marcados_1:
        cv2.circle(original_goi_1, (ponto[1], ponto[0]), 3, (0, 0, 255), -1)
        #original_goi_1[ponto[0]][ponto[1]] = [0, 0, 255]
    for ponto in pontos_marcados_2:
        cv2.circle(original_goi_2, (ponto[1], ponto[0]), 3, (0, 0, 255), -1)

    if printe:
        cv2.imshow('dst com tratamento '+imagem_utilizada+' 1', original_goi_1)
        cv2.imshow('dst com tratamento '+imagem_utilizada+' 2', original_goi_2)
        cv2.waitKey(0)


    cv2.imwrite(pasta_resultados+'dst com tratamento '+imagem_utilizada+' 1.jpg', original_goi_1)
    cv2.imwrite(pasta_resultados+'dst com tratamento '+imagem_utilizada+' 2.jpg', original_goi_2)
