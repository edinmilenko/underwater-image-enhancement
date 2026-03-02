import cv2
import numpy as np
import os
from contrast_tweaker import *
from denoising import *

if __name__ == "__main__":

    img_path_l = os.path.join(os.getcwd(), 'Input/left')
    img_path_r = os.path.join(os.getcwd(), 'Input/right')
    save_path_r = os.path.join(os.getcwd(), 'Output/right')
    save_path_l = os.path.join(os.getcwd(), 'Output/left')
   
    print(img_path_r)

    for image in os.listdir(img_path_l):

        #apply filters
        current_img = cv2.imread(os.path.join(img_path_l, image))
        laplacian_img = laplacian_sharpen(current_img)
        final_img = apply_CLAHE(laplacian_img)

        #save img
        cv2.imwrite(os.path.join(save_path_l, image), final_img)


