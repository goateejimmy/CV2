from PIL import Image
import numpy as np
import cv2
import os
def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    tiny_images_list = []
    for path in image_paths:
        image = cv2.imread(path)
        image=cv2.resize(image,dsize=(16,16), interpolation=cv2.INTER_NEAREST)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        image = np.array(image)
        feature = np.reshape(image,(1,256))
        feature = feature- np.average(feature)
        norm = np.linalg.norm(feature,ord =2,axis=1)
        feature = feature/norm
        
        
        tiny_images_list.append(feature)
    
    tiny_images = np.array(tiny_images_list)
    tiny_images = np.squeeze(tiny_images,axis=1)





    

    
    
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
