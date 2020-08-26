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


def calculaSSD(janela1, janela2):
    SSD = 100000
    try:
        SSD = sum(sum((np.array(janela1)-np.array(janela2))**2))

    except Exception as e:
        pass

    #print(SSD)
    if SSD == 0:
        return 0.1

    return SSD


def Casamento_pontos(pontos1, pontos2, imagem1, imagem2, Wssd, Tssd, Trazao):
    pontos_casados = []
    ssd_todos_pontos = []

    for ponto1 in pontos1:
        ssd_por_ponto = []
        janela1 = imagem1[(ponto1[0] - int(Wssd/2.)):(ponto1[0] + math.ceil(Wssd/2.)), (ponto1[1] - int(Wssd/2.)):(ponto1[1] + math.ceil(Wssd/2.))]
        for ponto2 in pontos2:
            janela2 = imagem2[(ponto2[0] - int(Wssd/2.)):(ponto2[0] + math.ceil(Wssd/2.)), (ponto2[1] - int(Wssd/2.)):(ponto2[1] + math.ceil(Wssd/2.))]
            ssd_por_ponto.append(calculaSSD(janela1, janela2))
        ssd_todos_pontos.append(ssd_por_ponto)

    pontos_escolhidos = []
    # print(ssd_todos_pontos)

    for i in range(len(pontos1)):
        #elimina pontos que possuem ssd maior que o limite
        if min(ssd_todos_pontos[i]) > Tssd:
            continue

        #elimina falsas correspondencias
        elif (sorted(ssd_todos_pontos[i])[0]/ sorted(ssd_todos_pontos[i])[1]) > Trazao:
            continue
        else:
            j = ssd_todos_pontos[i].index(sorted(ssd_todos_pontos[i])[0])
            if pontos2[j] not in pontos_escolhidos:
                pontos_escolhidos.append(pontos2[j])
                pontos_casados.append( [pontos1[i], pontos2[j]] )


    return pontos_casados


if __name__ == '__main__':

    #inicializacao de parametros
    Wd=3
    Wc=5*Wd
    k=0.04



    # Wh = 20
    # TR = 0.02
    # Wssd = 31
    # Tssd = 3500
    # Trazao = 0.9

    Whs = [25, 40, 50]
    TRs = [0.01, 0.02]
    Wssds = [21, 31]
    Tssds = [3000, 3500, 5000]
    Trazaos = [0.8, 0.9, 0.95]

    imagem_utilizada = "goi"
    pasta_resultados = "../results/casamento/"+imagem_utilizada+"/"

    #imagem
    imagem_goi_file_1 = "../dados/"+imagem_utilizada+"1.jpg"
    imagem_goi_file_2 = "../dados/"+imagem_utilizada+"2.jpg"


    for Wh in Whs:
        for TR in TRs:
            for Wssd in Wssds:
                for Tssd in Tssds:
                    for Trazao in Trazaos:


                        original_goi_1 = cv2.imread(imagem_goi_file_1, cv2.IMREAD_COLOR)
                        gray_goi_1 = cv2.cvtColor(original_goi_1, cv2.COLOR_BGR2GRAY)


                        original_goi_2 = cv2.imread(imagem_goi_file_2, cv2.IMREAD_COLOR)
                        gray_goi_2 = cv2.cvtColor(original_goi_2, cv2.COLOR_BGR2GRAY)

                        #cv2.imshow('original', original_goi_1)
                        #cv2.waitKey(0)


                        #############openc cv implementation

                        gray1 = np.float32(gray_goi_1)
                        dst1 = cv2.cornerHarris(gray1, Wc, Wd, k)

                        gray2 = np.float32(gray_goi_2)
                        dst2 = cv2.cornerHarris(gray2, Wc, Wd, k)



                        ##################my implementation

                        # dst = HarrisDetector(gray_goi_1, Wd, Wc, k)

                        #print(dst)

                        #Para Tr foi escolhido o valor 0.03 do valor maximo
                        marked_image_1 = original_goi_1.copy()
                        marked_image_1[dst1>TR*dst1.max()] = [0, 0, 255]

                        marked_image_2 = original_goi_2.copy()
                        marked_image_2[dst2>TR*dst2.max()] = [0, 0, 255]

                        #salva os pontos marcados
                        pontos_marcados_1 = []
                        pontos_marcados_2 = []

                        RED = np.array([0, 0, 255])

                        for y in range(marked_image_1.shape[0]):
                            for x in range(marked_image_1.shape[1]):
                                #print(marked_image[y][x])
                                if marked_image_1[y][x].any() == RED.any():
                                    pontos_marcados_1.append([y, x])

                        for y in range(marked_image_2.shape[0]):
                            for x in range(marked_image_2.shape[1]):
                                #print(marked_image[y][x])
                                if marked_image_2[y][x].any() == RED.any():
                                    pontos_marcados_2.append([y, x])

                        # print(pontos_marcados)

                        # cv2.imshow('dst com grupos 1', marked_image_1)
                        # cv2.waitKey(0)
                        #
                        # cv2.imshow('dst com grupos 2', marked_image_2)
                        # cv2.waitKey(0)

                        pontos_marcados_1 = AgrupaHarris(dst1, pontos_marcados_1, Wh)
                        pontos_marcados_2 = AgrupaHarris(dst2, pontos_marcados_2, Wh)

                        marked_image1 = original_goi_1.copy()
                        marked_image2 = original_goi_2.copy()


                        for ponto in pontos_marcados_1:
                            cv2.circle(marked_image1, (ponto[1], ponto[0]), 3, (0, 0, 255), -1)
                            #original_goi_1[ponto[0]][ponto[1]] = [0, 0, 255]

                        for ponto in pontos_marcados_2:
                            cv2.circle(marked_image2, (ponto[1], ponto[0]), 3, (0, 0, 255), -1)
                            #original_goi_2[ponto[0]][ponto[1]] = [0, 0, 255]

                        #cv2.imshow('dst com tratamento 1', marked_image1)

                        #cv2.imshow('dst com tratamento 2', marked_image2)
                        #cv2.waitKey(0)


                        pontos_casados = Casamento_pontos(pontos_marcados_1, pontos_marcados_2, gray_goi_1, gray_goi_2, Wssd, Tssd, Trazao)

                        #print(pontos_casados)
                        #gera imagem com as 2 figuras e faz uma linha entre os pontos casados

                        imagem_combinada = np.concatenate((marked_image1, marked_image2), axis = 1)

                        for par in pontos_casados:
                            par[1][1] = par[1][1]+original_goi_1.shape[1]
                            imagem_combinada = cv2.line(imagem_combinada, tuple(par[0][::-1]), tuple(par[1][::-1]), (0, 255, 0), 1)



                        # cv2.imshow('imagem concatenada', imagem_combinada)
                        # cv2.waitKey(0)

                        cv2.imwrite(pasta_resultados+'imagem_concatenada_'+imagem_utilizada+'_'+str(Wh)+'_'+str(TR)+'_'+str(Wssd)+'_'+str(Tssd)+'_'+str(Trazao)+'.png', imagem_combinada)
