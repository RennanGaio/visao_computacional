import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import os
import pandas as pd
from scipy import linalg
import pysift
from scipy.linalg import null_space
from scipy.optimize import least_squares

#funções do metodo dlt para encontrar projeções

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


def fundamental_from_projections(P1,P2):

	F = [[0,0,0],[0,0,0],[0,0,0]]

	X = [0,0,0]

	X[0] = np.vstack(( P1[1], P1[2] ))
	X[1] = np.vstack(( P1[2], P1[0] ))
	X[2] = np.vstack(( P1[0], P1[1] ))

	Y = [0,0,0]

	Y[0] = np.vstack(( P2[1], P2[2] ))
	Y[1] = np.vstack(( P2[2], P2[0] ))
	Y[2] = np.vstack(( P2[0], P2[1] ))


	for i in range(3):
		for j in range(3):
			XY = np.vstack((X[j], Y[i]))
			F[i][j] = np.linalg.det(XY)


	return F


def calculate_camera_center(M):
	
	Q=M[0:3,0:3]
	Qinv=np.linalg.inv(Q)
	cc=np.matmul(-Qinv,M[:,3])
	
	return cc

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
    
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])

        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def compute_fundamental_8_point_sem_rank(pts1, pts2):
	'''Computes the fundamental matrix from corresponding points x1, x2 using
	the 8 point algorithm.'''
	n = len(pts1)
    
    
	if len(pts2) != n:
		raise ValueError('Number of points do not match.')

	x1 = []
	x2 = []

	for i in range(n):
		x1.append(np.concatenate((pts1[i], [1])))
		x2.append(np.concatenate((pts2[i], [1])))

	x1 = np.float64(x1)
	x2 = np.float64(x2)


	A = np.zeros((n, 9))
    
    
	for i in range(n):   
		A[i] = np.kron(x2[i], x1[i])


	# Solve A*f = 0 using least squares.
	U, S, V = np.linalg.svd(A)
	F = V[-1].reshape(3, 3)

	return F / F[2][2]


def compute_fundamental_8_point(pts1, pts2):
	'''Computes the fundamental matrix from corresponding points x1, x2 using
	the 8 point algorithm.'''
	n = len(pts1)
    
    
	if len(pts2) != n:
		raise ValueError('Number of points do not match.')

	x1 = []
	x2 = []

	for i in range(n):
		x1.append(np.concatenate((pts1[i], [1])))
		x2.append(np.concatenate((pts2[i], [1])))

	x1 = np.float64(x1)
	x2 = np.float64(x2)


	A = np.zeros((n, 9))
    
    
	for i in range(n):   
		A[i] = np.kron(x2[i], x1[i])


	# Solve A*f = 0 using least squares.
	U, S, V = np.linalg.svd(A)
	F = V[-1].reshape(3, 3)

	# Constrain F to rank 2 by zeroing out last singular value.
	U, S, V = np.linalg.svd(F)
	S[2] = 0
	F = np.dot(U, np.dot(np.diag(S), V))
	return F / F[2][2]

def normalize(points):
	n = len(points)
	x1 = []

	for i in range(n):
		x1.append(np.concatenate((points[i], [1])))

	points = np.asarray(x1)

	m = np.mean(points, 0)
	s = np.std(points)

	T = np.array([[s, 0, m[0]], [0, s, m[1]], [0, 0, 1]])

	normalized_points = np.dot( T, points.T )
	normalized_points = normalized_points.T

	normalized_points = normalized_points[:, 0:2]
	
	T = np.linalg.inv(T)

	return T, normalized_points

def normalize2(points):
	N=points.shape[0]

	m=np.average(points,axis=0)

	m_mean=points-m.reshape(1,2)

	s_sum=np.sum((m_mean)**2,axis=None)

	s=(s_sum/(2*N))**0.5

	sinv=1/s

	x=m_mean*sinv

	T = np.zeros((3,3))
	T[0,0] = sinv
	T[1,1] = sinv
	T[2,2] = 1
	T[0,2] = -sinv*m[0]
	T[1,2] = -sinv*m[1]

	return T, x

def compute_fundamental_8_point_normalized(x1, x2):
	'''Computes the fundamental matrix from corresponding points x1, x2 using
	the normalized 8 point algorithm.'''
	n = x1.shape[1]
	if x2.shape[1] != n:
		raise ValueError('Number of points do not match.')

	# normalize.
	T1, x1 = normalize2(x1)
	T2, x2 = normalize2(x2)

	F = compute_fundamental_8_point(x1, x2)

	# denormalize.
	# F = np.dot(T1.T, np.dot(F, T2))
	F = np.dot(T2.T, np.dot(F, T1))

	return F / F[2][2]


def automatic_fundamental_8_points_sem_rank(pts1, pts2):
	#uses ransac to compute the 8 point
	N=15000
	S=pts1.shape[0]
	r=np.random.randint(S,size=(N,8))

	m=np.ones((3,S))
	m[0:2,:]=pts1.T
	mdash=np.ones((3,S))
	mdash[0:2,:]=pts2.T
	count=np.zeros(N)
	cost=np.zeros(S)
	t=1e-2
	for i in range(N):
		cost1=np.zeros(8)
		F=compute_fundamental_8_point_sem_rank(pts1[r[i,:],:],pts2[r[i,:],:])
		if len(F):
			for j in range(S):
				cost[j]=np.dot(np.dot(mdash[:,j].T,F),m[:,j])
			inlie=np.absolute(cost)<t
			count[i]=np.sum(inlie + np.zeros(S),axis=None)

	index=np.argsort(-count)
	best=index[0]
	best_F=compute_fundamental_8_point_sem_rank(pts1[r[best,:],:],pts2[r[best,:],:])
	for j in range(S):
		cost[j]=np.dot(np.dot(mdash[:,j].T,best_F),m[:,j])
	confidence=np.absolute(cost)
	index=np.argsort(confidence)
	pts2=pts2[index]
	pts1=pts1[index]

	inliers1=pts1[:100,:]
	inliers2=pts2[:100,:]

	return best_F, inliers1, inliers2

def automatic_fundamental_8_points(pts1, pts2):
	#uses ransac to compute the 8 point
	N=15000
	S=pts1.shape[0]
	r=np.random.randint(S,size=(N,8))

	m=np.ones((3,S))
	m[0:2,:]=pts1.T
	mdash=np.ones((3,S))
	mdash[0:2,:]=pts2.T
	count=np.zeros(N)
	cost=np.zeros(S)
	t=1e-2
	for i in range(N):
		cost1=np.zeros(8)
		F=compute_fundamental_8_point(pts1[r[i,:],:],pts2[r[i,:],:])
		if len(F):
			for j in range(S):
				cost[j]=np.dot(np.dot(mdash[:,j].T,F),m[:,j])
			inlie=np.absolute(cost)<t
			count[i]=np.sum(inlie + np.zeros(S),axis=None)

	index=np.argsort(-count)
	best=index[0]
	best_F=compute_fundamental_8_point(pts1[r[best,:],:],pts2[r[best,:],:])
	for j in range(S):
		cost[j]=np.dot(np.dot(mdash[:,j].T,best_F),m[:,j])
	confidence=np.absolute(cost)
	index=np.argsort(confidence)
	pts2=pts2[index]
	pts1=pts1[index]

	inliers1=pts1[:100,:]
	inliers2=pts2[:100,:]

	return best_F, inliers1, inliers2

def automatic_fundamental_8_points_norm(pts1, pts2):

	#uses ransac to compute the 8 point
	N=15000
	S=pts1.shape[0]
	r=np.random.randint(S,size=(N,8))

	m=np.ones((3,S))
	m[0:2,:]=pts1.T
	mdash=np.ones((3,S))
	mdash[0:2,:]=pts2.T
	count=np.zeros(N)
	cost=np.zeros(S)
	t=1e-2
	for i in range(N):
		cost1=np.zeros(8)
		F=compute_fundamental_8_point_normalized(pts1[r[i,:],:],pts2[r[i,:],:])
		if len(F):
			for j in range(S):
				cost[j]=np.dot(np.dot(mdash[:,j].T,F),m[:,j])
			inlie=np.absolute(cost)<t
			count[i]=np.sum(inlie + np.zeros(S),axis=None)

	index=np.argsort(-count)
	best=index[0]
	best_F=compute_fundamental_8_point_normalized(pts1[r[best,:],:],pts2[r[best,:],:])
	for j in range(S):
		cost[j]=np.dot(np.dot(mdash[:,j].T,best_F),m[:,j])
	confidence=np.absolute(cost)
	index=np.argsort(confidence)
	pts2=pts2[index]
	pts1=pts1[index]

	inliers1=pts1[:100,:]
	inliers2=pts2[:100,:]

	return best_F, inliers1, inliers2


def automatic_fundamental_8_points_gold(pts1, pts2):

	#uses ransac and gold standard method to compute the 8 point
	N=15000
	S=pts1.shape[0]
	r=np.random.randint(S,size=(N,8))

	m=np.ones((3,S))
	m[0:2,:]=pts1.T
	mdash=np.ones((3,S))
	mdash[0:2,:]=pts2.T
	count=np.zeros(N)
	cost=np.zeros(S)
	t=1e-2
	for i in range(N):
		cost1=np.zeros(8)
		F=compute_fundamental_8_point_normalized(pts1[r[i,:],:],pts2[r[i,:],:])

		#roda o gold standard na primeira iteração
		if i==0:
			ponto_epipolar = null_space(F)[:,0:1]

			ex = [[0, -ponto_epipolar[2], ponto_epipolar[1]],[ponto_epipolar[2], 0, -ponto_epipolar[0]],[-ponto_epipolar[1], ponto_epipolar[0], 0]]

			P2 = np.hstack((ex@F, ponto_epipolar))
			P1 = np.hstack((np.eye(3, 3), np.zeros((3, 1))))

			point_4d_hom = cv2.triangulatePoints(P1, P2, np.expand_dims(pts1[r[i,:],:], axis=1), np.expand_dims(pts2[r[i,:],:], axis=1))
			point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
			

			x_hat1 = P1@point_4d
			x_hat2 = P2@point_4d

			res_1 = least_squares(fun_rosenbrock, x0_rosenbrock)

		if len(F):
			for j in range(S):
				cost[j]=np.dot(np.dot(mdash[:,j].T,F),m[:,j])
			inlie=np.absolute(cost)<t
			count[i]=np.sum(inlie + np.zeros(S),axis=None)

	index=np.argsort(-count)
	best=index[0]
	best_F=compute_fundamental_8_point_normalized(pts1[r[best,:],:],pts2[r[best,:],:])
	for j in range(S):
		cost[j]=np.dot(np.dot(mdash[:,j].T,best_F),m[:,j])
	confidence=np.absolute(cost)
	index=np.argsort(confidence)
	pts2=pts2[index]
	pts1=pts1[index]

	inliers1=pts1[:100,:]
	inliers2=pts2[:100,:]

	return best_F, inliers1, inliers2


def calculate_SIFT(img1, img2):
	kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
	kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)

	# FLANN parameters
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)

	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)

	good = []
	pts1 = []
	pts2 = []

	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
	    if m.distance < 0.8*n.distance:
	        good.append(m)
	        pts2.append(kp2[m.trainIdx].pt)
	        pts1.append(kp1[m.queryIdx].pt)
	        
	pts1 = np.int32(pts1)
	pts2 = np.int32(pts2)

	return pts1, pts2