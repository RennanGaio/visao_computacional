# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os


def distancia_2_pontos(p1, p2):
    return math.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )

def linha_a_partir_2_pontos(X, Y):

    linha_homogenea=np.cross(X,Y)
    return linha_homogenea

def intersecao_2_linhas(L1, L2):
    ponto_intersecao = np.cross(L1, L2)
    ponto_intersecao=ponto_intersecao/ponto_intersecao[2]
    return ponto_intersecao

def cria_linhas(pontos):
    linhas = []
    for i in range(int(len(pontos)/2)):
        linhas.append(linha_a_partir_2_pontos(pontos[2*i], pontos[2*i+1]))

    return linhas


def cria_pontos_fulga(linhas):
    pontos=[]
    for i in range(int(len(linhas)/2)):
        pontos.append(intersecao_2_linhas(linhas[2*i], linhas[2*i+1]))
    return pontos

def desenha_linha(imagem, linha_homogenea):
    #np.linspace(0, imagem.shape[1]*10, 10)
    X=[0, imagem.shape[1]]
    Y=[]

    for e in X:
        Y.append(int(-(e*linha_homogenea[0]+linha_homogenea[2])/linha_homogenea[1]))

    print(Y)

    cv2.line(imagem, (X[0], Y[0]), (X[1], Y[1]), (0, 0, 255), 3 )
    return imagem

def transformacao_projetiva(imagem_projetiva, imagem_auxiliar, linha_do_infinito):
    matriz_projetiva = [[1, 0, 0],[0, 1, 0], [linha_do_infinito[0], linha_do_infinito[1], linha_do_infinito[2]]]
    #matriz_projetiva.append(linha_do_infinito)

    print(matriz_projetiva)

    hinv = np.linalg.inv(matriz_projetiva)
    htinv = np.transpose(hinv)

    for y in range(imagem_projetiva.shape[0]): #percorre linhas da imagem original
        for x in range(imagem_projetiva.shape[1]): #percorre colunas da imagem original
            posicao_final=np.dot(htinv, [x, y, 1])
            if posicao_final[0]>=0 and posicao_final[0]<imagem_auxiliar.shape[1] and posicao_final[1]>=0 and posicao_final[1]<imagem_auxiliar.shape[0]:
                imagem_auxiliar[int(posicao_final[1]), int(posicao_final[0])] = imagem_projetiva[y,x]


    return imagem_auxiliar


def traca_linhas(imagem):
    #traça primeira linha (la)
    cv2.line(imagem, (161, 475), (286, 112), (0, 0, 255), 3 )
    #traça segunda linha (lb)
    cv2.line(imagem, (704, 501), (646, 211), (0, 0, 255), 3 )
    #traça terceira linha (lc)
    cv2.line(imagem, (186, 151), (359, 155), (0, 0, 255), 3 )
    #traça quarta linha (ld)
    cv2.line(imagem, (75, 385), (314, 394), (0, 0, 255), 3 )

    return imagem

def marca_pontos(imagem, lista_de_pontos):
    for p in lista_de_pontos:
        cv2.circle(imagem, (p[0], p[1]), 3, (0, 255, 0), -1)
    return imagem

#ordem das cores na imagem
#(b, g, r) = imagem[0,0]

#ordem das coordenadas
#shape[0] é a altura, eixo y
#shape[1] é a largura, eixo x
if __name__ == '__main__':

    #Carrega a imagem
    nome_da_imagem="imagens/chao3.png"

    imagem = cv2.imread(nome_da_imagem, cv2.IMREAD_COLOR)
    print("shape da imagem")
    print(imagem.shape)


    imagem_com_espaco = np.zeros((1500, 1500, 3), dtype=np.uint8)

    cv2.imshow("original", imagem)
    cv2.waitKey(0)

    rows, cols, channels = imagem.shape
    imagem_com_espaco[1500-rows:1500, 1500-cols:1500] = imagem

    pontos=[[161, 475, 1],
    [646, 211, 1],
    [314, 394, 1]]


    #cria imagem com linhas marcando retas paralelas no mundo real
    imagem=marca_pontos(imagem, pontos)

    cv2.imshow("original", imagem)

    cv2.waitKey(0)


    d1 = distancia_2_pontos(pontos[0], pontos[2])
    d2 = distancia_2_pontos(pontos[2], pontos[1])

    razao_de_distancias_na_imagem = d1/d2
    razao_de_distancias_real = 1./3. 




    '''##################################################################################################
    ##################################################################################################

    #Cria imagem para reta no infinito




    #nesse vetor de linhas sao paralelas as linhas 2*i e 2*i+1 respectivamente. ex. linha[0] e paralela a linha [1], linha[2] e paralela a linha[3]
    linhas=cria_linhas(pontos)

    pontos_de_fulga=cria_pontos_fulga(linhas)

    print("vetor de pontos de fulga em coordenadas homogeneas")
    print(pontos_de_fulga)


    #cria linha do infinito a partir dos 2 pontos de fulga solicitados
    linha_do_infinito=linha_a_partir_2_pontos(pontos_de_fulga[0], pontos_de_fulga[1])


    imagem_linha_infinito=desenha_linha(imagem_linha_infinito, linha_do_infinito)

    #cv2.imshow("linha do infinito", imagem_linha_infinito)

    #cv2.waitKey(0)


    ##################################################################################################
    ##################################################################################################
    #Cria imagem para transformacao projetiva

    #Cria imagem para reta no infinito
    pontos=[[161, 475, 1],
    [286, 112, 1],
    [704, 501, 1],
    [646, 211, 1],
    [186, 151, 1],
    [359, 155, 1],
    [75, 385, 1],
    [314, 394, 1]]

    #ajusta as coordenadas a imagem alterada com a parte preta, para tracar linha no infinito

    for p in pontos:
        p[0]+=100
        p[1]+=100

    print(pontos)

    #nesse vetor de linhas sao paralelas as linhas 2*i e 2*i+1 respectivamente. ex. linha[0] e paralela a linha [1], linha[2] e paralela a linha[3]
    linhas=cria_linhas(pontos)

    pontos_de_fulga=cria_pontos_fulga(linhas)

    #cria linha do infinito a partir dos 2 pontos de fulga solicitados
    linha_do_infinito=linha_a_partir_2_pontos(pontos_de_fulga[0], pontos_de_fulga[1])

    #print(linha_do_infinito)

    #normaliza vetor
    #for i in range(len(linha_do_infinito)):
    #    linha_do_infinito[i]=linha_do_infinito[i]/np.sum(linha_do_infinito)




    imagem_projetiva = np.zeros((800, 1000, 3), dtype=np.uint8)
    rows, cols, channels = imagem.shape
    imagem_projetiva[100:100+rows, 100:100+cols] = imagem

    cv2.imshow("original", imagem_projetiva)
    #plt.imshow(imagem_projetiva)

    cv2.waitKey(0)

    imagem_auxiliar = np.zeros((800, 1000, 3), dtype=np.uint8)

    imagem_auxiliar = transformacao_projetiva(imagem_projetiva, imagem_auxiliar, linha_do_infinito)

    cv2.imshow("transformacao projetiva", imagem_auxiliar)
    #plt.imshow(imagem_auxiliar)

    #plt.show()

    cv2.waitKey(0)'''

    cv2.destroyAllWindows()
