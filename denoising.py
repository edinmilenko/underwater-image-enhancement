import cv2
import os
from utils import *

# higher kernel for higher noise (it must be 3 < kernel < 7)
def median_blur(kernel):

    if kernel > 7 or kernel < 3:
        raise ValueError("Value is invalid")

    input_path = os.path.join("data", "raw")
    o_path = os.path.join(os.getcwd(), "data", "processed")
    image_list = list_images(input_path)

    for image in image_list:
        img = cv2.imread(image)
        denoised = cv2.medianBlur(img, kernel)
        filename = os.path.basename(image)
        output_path = os.path.join(o_path, filename)
        save_image(output_path, denoised)


def bilateral_filter():

    input_path = os.path.join("data", "raw")
    o_path = os.path.join(os.getcwd(), "data", "processed1")
    image_list = list_images(input_path)

    for image in image_list:
        img = cv2.imread(image)
        denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        filename = os.path.basename(image)
        output_path = os.path.join(o_path, filename)
        save_image(output_path, denoised)

def non_local_means_denoising():

    input_path = os.path.join("data", "raw")
    o_path = os.path.join(os.getcwd(), "data", "processed2")
    image_list = list_images(input_path)

    for image in image_list:
        img = cv2.imread(image)
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        filename = os.path.basename(image)
        output_path = os.path.join(o_path, filename)
        save_image(output_path, denoised)



if __name__ == "__main__":
    non_local_means_denoising()