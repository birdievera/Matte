from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import scipy as sp
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image
import time

def alpha_composite(compA, backA, compB, backB):
    '''
    Return the alpha composite, and foregrounds given
    two composite photos and their backgrounds
    
    Takes: compA, compB (foreground object against diff background),
    backA, backB (just background)
    Returns: Boolean based on success of algorithm
    
    Rewrite the matting equation to:
    Ax = b form
    
    [1 0 0 -B1_r]           [C1_r - B1_r]
    [0 1 0 -B1_g] [ Fr  ]   [C1_g - B1_g]
    [0 0 1 -B1_b] [ Fg  ] = [C1_b - B1_b]
    [1 0 0 -B2_r] [ Fb  ]   [C2_r - B2_r]
    [0 1 0 -B2_g] [alpha]   [C2_g - B2_g]
    [0 0 1 -B3_b]           [C2_b - B2_b]
    
    
    Then solve the algorithm by doing the pseudoinverse of A
    and doing the dot product of that with b.
    
    Takes: Approx 13 seconds for a large photo
    '''
    
    foreground = np.zeros((compA.shape[0], compA.shape[1], 3L))
    alpha = np.zeros((compA.shape[0], compA.shape[1]))  
    
    #for every pixel, apply the matting formula
    for i in range (compA.shape[0]):
        for j in range (compA.shape[1]):
            
            a = np.array([[1,0,0, backA[i,j,0]*(-1)],
                       [0,1,0, backA[i,j,1]*(-1)],
                       [0,0,1, backA[i,j,2]*(-1)],
                       [1,0,0, backB[i,j,0]*(-1)],
                       [0,1,0, backB[i,j,1]*(-1)],
                       [0,0,1, backB[i,j,2]*(-1)]])
            
            pseudoinverse = np.linalg.pinv(a)
            
            b = np.array([[compA[i,j,0]-backA[i,j,0]],
                  [compA[i,j,1]-backA[i,j,1]],
                  [compA[i,j,2]-backA[i,j,2]],
                  [compB[i,j,0]-backB[i,j,0]],
                  [compB[i,j,1]-backB[i,j,1]],
                  [compB[i,j,2]-backB[i,j,2]]])

            x = np.dot(pseudoinverse, b)
            
            # clip to make sure it's in proper colour range [0, 1] since dot
            # will result in higher numbers
            x = np.clip(x, 0, 1)

            
            # resulting x is in the form of:
            # [ Fr  ]
            # [ Fg  ]
            # [ Fb  ]
            # [alpha]
            
            # get alpha
            alpha[i][j] = x[3][0]
            
            # get foreground (rgb colours)
            foreground[i][j] = np.array([x[0][0], x[1][0], x[2][0]])
            
    return foreground, alpha

def alpha_composite2 (compA, backA, compB, backB):
    '''
    Uses least square solving.
    Much slower than the 1st function.
    
    Takes: Approx 17 seconds for a large picture
    '''
    
    foreground = np.zeros((compA.shape[0], compA.shape[1], 3L))
    alpha = np.zeros((compA.shape[0], compA.shape[1]))
    
    #for every pixel, apply the matting formula
    for i in range (compA.shape[0]):
        for j in range (compA.shape[1]):
            
            a = np.array([[1,0,0, backA[i,j,0]*(-1)],
                       [0,1,0, backA[i,j,1]*(-1)],
                       [0,0,1, backA[i,j,2]*(-1)],
                       [1,0,0, backB[i,j,0]*(-1)],
                       [0,1,0, backB[i,j,1]*(-1)],
                       [0,0,1, backB[i,j,2]*(-1)]])
            
            b = np.array([[compA[i,j,0]-backA[i,j,0]],
                  [compA[i,j,1]-backA[i,j,1]],
                  [compA[i,j,2]-backA[i,j,2]],
                  [compB[i,j,0]-backB[i,j,0]],
                  [compB[i,j,1]-backB[i,j,1]],
                  [compB[i,j,2]-backB[i,j,2]]])
  
            x = np.linalg.lstsq(a,b)[0]  
            
            # clip to make sure it's in proper colour range [0, 1] since dot
            # will result in higher numbers
            x = np.clip(x, 0, 1)

            
            # resulting x is in the form of:
            # [ Fr  ]
            # [ Fg  ]
            # [ Fb  ]
            # [alpha]
            
            # get alpha
            alpha[i][j] = x[3][0]
            
            # get foreground (rgb colours)
            foreground[i][j] = np.array([x[0][0], x[1][0], x[2][0]])
            
    return foreground, alpha    

def alpha_composite3 (compA, backA, compB, backB):
    '''
    Uses qr decomposition in order to solve matting
    Much slower than the 1st function
    
    Takes: Approx 22 seconds
    '''
    
    foreground = np.zeros((compA.shape[0], compA.shape[1], 3L))
    alpha = np.zeros((compA.shape[0], compA.shape[1]))    
    
    #for every pixel, apply the matting formula
    for i in range (compA.shape[0]):
        for j in range (compA.shape[1]):
            
            a = np.array([[1,0,0, backA[i,j,0]*(-1)],
                       [0,1,0, backA[i,j,1]*(-1)],
                       [0,0,1, backA[i,j,2]*(-1)],
                       [1,0,0, backB[i,j,0]*(-1)],
                       [0,1,0, backB[i,j,1]*(-1)],
                       [0,0,1, backB[i,j,2]*(-1)]])
            
            b = np.array([[compA[i,j,0]-backA[i,j,0]],
                  [compA[i,j,1]-backA[i,j,1]],
                  [compA[i,j,2]-backA[i,j,2]],
                  [compB[i,j,0]-backB[i,j,0]],
                  [compB[i,j,1]-backB[i,j,1]],
                  [compB[i,j,2]-backB[i,j,2]]])
            
            # qr decomposition of A
            
            Q, R = np.linalg.qr(a) 
            
            # computing Q^T*b (project b onto the range of A)
            Qb = np.dot(Q.T,b) 
            
            # solving R*x = Q^T*b  
            x = sp.linalg.solve_triangular(R, Qb, check_finite=False)          
            
            # clip to make sure it's in proper colour range [0, 1] since dot
            # will result in higher numbers
            x = np.clip(x, 0, 1)

            
            # resulting x is in the form of:
            # [ Fr  ]
            # [ Fg  ]
            # [ Fb  ]
            # [alpha]
            
            # get alpha
            alpha[i][j] = x[3][0]
            
            # get foreground (rgb colours)
            foreground[i][j] = np.array([x[0][0], x[1][0], x[2][0]])
            
    return foreground, alpha


def matte (cA_r, cA_g, cA_b, cB_r, cB_g, cB_b, 
          bA_r, bA_g, bA_b, bB_r, bB_g, bB_b):
    A = (np.linalg.pinv(np.array([[1,0,0, bA_r*(-1)],
                                  [0,1,0, bA_g*(-1)],
                                  [0,0,1, bA_b*(-1)],
                                  [1,0,0, bB_r*(-1)],
                                  [0,1,0, bB_g*(-1)],
                                  [0,0,1, bB_b*(-1)]])))
    b = np.array([[cA_r-bA_r],
                  [cA_g-bA_g],
                  [cA_b-bA_b],
                  [cB_r-bB_r],
                  [cB_g-bB_g],
                  [cB_b-bB_b]])
    return np.dot(A, b)

def alpha_composite4 (compA, backA, compB, backB):
    '''
    Vectorize and map the function
    
    Approx 12.7 seconds
    '''
    
    foreground = np.zeros((compA.shape[0], compA.shape[1], 3L))
    alpha = np.zeros((compA.shape[0], compA.shape[1]))
    
    find_matte = np.vectorize(matte, otypes = [np.ndarray])
        
    x = find_matte(compA[:,:,0], compA[:,:,1], compA[:,:,2], 
                   compB[:,:,0], compB[:,:,1], compB[:,:,2],
                   backA[:,:,0], backA[:,:,1], backA[:,:,2], 
                   backB[:,:,0], backB[:,:,1], backB[:,:,2])
    
    x = x.reshape((1, x.size))

    # get alpha
    alpha += np.array([i[3][0] for i in x[0]]).reshape(alpha.shape[:2])
    
    # get foreground (rgb colours)
    foreground[:,:,0] += np.array([i[0][0] for i in x[0]]).reshape(alpha.shape[:2])
    foreground[:,:,1] += np.array([i[1][0] for i in x[0]]).reshape(alpha.shape[:2])
    foreground[:,:,2] += np.array([i[2][0] for i in x[0]]).reshape(alpha.shape[:2])
    
    foreground = np.clip(foreground, 0, 1)
    alpha = np.clip(alpha, 0, 1)
            
    return foreground, alpha

def alpha_composite5 (compA, backA, compB, backB):
    '''
    Generalization of the formula: one big sum of the pictures
    
    Approx 0.5 seconds for a large photo
    
    Fastest algorithm
    '''
    
    alpha = np.clip(1-(np.sum((compA-compB)*(backA-backB), axis=2)/np.sum((backA-backB)**2, axis=2)), 0, 1)
    
    foreground = np.clip(((compB+compA)/2 - (np.dstack(((1-alpha), (1-alpha), (1-alpha))) * (backB+compA)/2)), 0, 1)
    
    return foreground, alpha

def create_composite(alphaIn, colIn, backIn):
    '''
    Creates a new image by applying the alpha channel and adding the foreground
    '''
    alphaIn = np.dstack((alphaIn, alphaIn, alphaIn))
    
    # From C = Co + (1-alpha)Ck 
    compOut = colIn + ((1.0 - alphaIn) * backIn)
    
    return compOut
    
if __name__ == "__main__":  
    
    im_matrix = []
    # flower photo A and background of it
    
    C1 = imread(os.getcwd()+"/imgs/flowers-compA.jpg")/255.0
    B1 = imread(os.getcwd()+"/imgs/flowers-backA.jpg")/255.0
    
    # flower photo B and background of it
    
    C2 = imread(os.getcwd()+"/imgs/flowers-compB.jpg")/255.0
    B2 = imread(os.getcwd()+"/imgs/flowers-backB.jpg")/255.0 
    
    #background image
    
    background = imread(os.getcwd()+"/imgs/window.jpg")/255.0
    
    start = time.time() 
    
    foreground, alpha = alpha_composite5(C1,B1,C2,B2)
    
    print 'It took', time.time()-start, 'seconds.' 
    
    imsave('flowersA-alpha.jpg', alpha, cmap=cm.gray)
    imsave("flowersA-foreground.jpg", foreground)
    
    # composite
    
    newbg = imread(os.getcwd()+"/imgs/window.jpg")/255.0
    
    composite = create_composite(alpha, foreground, newbg)
    
    imsave('flowers-composite.jpg', composite)
    