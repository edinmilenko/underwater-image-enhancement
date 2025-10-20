import cv2
import numpy as np
import os
from contrast_tweaker import *
from denoising import *

def process_video(input_path, output_path):

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Errore nell'apertura del video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Premi 'q' per interrompere l'esecuzione.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # fine video
        polished = bilateral_filter(frame)
        enhanced = laplacian_sharpen(apply_CLAHE(polished))      
        combined = np.hstack((frame, enhanced))
        cv2.imshow("Originale (sinistra)  |  CLAHE (destra)", combined)

        out.write(enhanced)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "piedi.mp4"
    output_video = "left_clahe.mp4" 

    process_video(video_path, output_video)
