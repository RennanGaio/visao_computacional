# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os
from sklearn.metrics import mean_squared_error
import pysift



if __name__ == '__main__':

    imagens_utilizadas = ["goi", "tajnew", "tajold"]
    for imagem_utilizada in imagens_utilizadas:


        pasta_resultados = "../results/ransac/"+imagem_utilizada+"/"

        #imagem
        imagem_file_1 = "../dados/"+imagem_utilizada+"1.jpg"
        imagem_file_2 = "../dados/"+imagem_utilizada+"2.jpg"


        original_image_1 = cv2.imread(imagem_file_1, cv2.IMREAD_COLOR)
        gray_image_1 = cv2.cvtColor(original_image_1, cv2.COLOR_BGR2GRAY)


        original_image_2 = cv2.imread(imagem_file_2, cv2.IMREAD_COLOR)
        gray_image_2 = cv2.cvtColor(original_image_2, cv2.COLOR_BGR2GRAY)

        #cv2.imshow('original', original_goi_1)
        #cv2.waitKey(0)

        # Compute SIFT keypoints and descriptors
        key_points_1, des1 = pysift.computeKeypointsAndDescriptors(gray_image_1)
        key_points_2, des2 = pysift.computeKeypointsAndDescriptors(gray_image_2)

        pontos_marcados_1 = np.int32([point.pt for point in key_points_1])
        pontos_marcados_2 = np.int32([point.pt for point in key_points_2])


        #print(pontos_marcados_1)


        #Cria imagens auxiliares para plot
        marked_image1 = original_image_1.copy()
        marked_image2 = original_image_2.copy()



        for ponto in pontos_marcados_1:
            cv2.circle(marked_image1, (ponto[0], ponto[1]), 3, (0, 0, 255), -1)
            #original_goi_1[ponto[0]][ponto[1]] = [0, 0, 255]

        for ponto in pontos_marcados_2:
            cv2.circle(marked_image2, (ponto[0], ponto[1]), 3, (0, 0, 255), -1)
            #original_goi_2[ponto[0]][ponto[1]] = [0, 0, 255]

        #plota imagens marcadas com os pontos identificados

        # cv2.imshow('key points SIFT imagem 1', marked_image1)
        #
        # cv2.imshow('key points SIFT imagem 2', marked_image2)
        # cv2.waitKey(0)

        cv2.imwrite(pasta_resultados+'SIFT_key_points_'+imagem_utilizada+'_1.jpg', marked_image1)
        cv2.imwrite(pasta_resultados+'SIFT_key_points_'+imagem_utilizada+'_2.jpg', marked_image2)


        #casamento de caracteristicas

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good = []
        for p1, p2 in matches:
            if p1.distance < 0.7 * p2.distance:
                good.append(p1)

        # Estimate homography between template and scene
        src_pts = np.float32([ key_points_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([ key_points_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        #com ransac
        H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        print(H)

        imagem_combinada = np.concatenate((original_image_1, original_image_2), axis = 1)

        # Draw SIFT keypoint matches
        for p in good:
            pt1 = (int(key_points_1[p.queryIdx].pt[0]), int(key_points_1[p.queryIdx].pt[1]))
            pt2 = (int(key_points_2[p.trainIdx].pt[0]  + original_image_1.shape[1] ), int(key_points_2[p.trainIdx].pt[1]))
            cv2.line(imagem_combinada, pt1, pt2, (0, 255, 0))

        # cv2.imshow('imagem concatenada', imagem_combinada)
        # cv2.waitKey(0)

        cv2.imwrite(pasta_resultados+'correspondencia_SIFT_'+imagem_utilizada+'.jpg', imagem_combinada)


        #transforma imagem 1 com a homografia H retornada do casamento dos pontos

        homografied_image_1 = cv2.warpPerspective(original_image_1, H, (int(original_image_1.shape[1]*1.5), int(original_image_1.shape[0]*1.5)))

        # cv2.imshow('homografia imagem 1 sem ransac', homografied_image_1)
        # cv2.waitKey(0)

        cv2.imwrite(pasta_resultados+'homografia_SIFT_all_points_'+imagem_utilizada+'.jpg', homografied_image_1)
