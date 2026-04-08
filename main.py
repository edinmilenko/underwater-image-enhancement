import cv2
import numpy as np
import os
import torch
from contrast_tweaker import apply_CLAHE, laplacian_sharpen
from denoising import *

from waternet.net import WaterNet
from waternet.data import transform


def arr2ten(arr):
    ten = torch.from_numpy(arr) / 255
    if len(ten.shape) == 3:
        ten = torch.permute(ten, (2, 0, 1))
        ten = torch.unsqueeze(ten, dim=0)
    elif len(ten.shape) == 4:
        ten = torch.permute(ten, (0, 3, 1, 2))
    return ten

def ten2arr(ten):
    arr = ten.cpu().detach().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    arr = np.transpose(arr, (0, 2, 3, 1))
    return arr

def preprocess(rgb_arr):
    wb, gc, he = transform(rgb_arr)
    rgb_ten = arr2ten(rgb_arr)
    wb_ten  = arr2ten(wb)
    gc_ten  = arr2ten(gc)
    he_ten  = arr2ten(he)
    return rgb_ten, wb_ten, he_ten, gc_ten

def postprocess(model_out):
    return ten2arr(model_out)


def load_waternet():
    weights_path = os.path.join(os.getcwd(), "weights.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
        )

    model = WaterNet()
    ckpt = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(ckpt)
    model.eval()
    return model


def apply_waternet(image_bgr, model):
    rgb_im = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb_ten, wb_ten, he_ten, gc_ten = preprocess(rgb_im)

    with torch.no_grad():
        out_ten = model(rgb_ten, wb_ten, he_ten, gc_ten)

    out_im = postprocess(out_ten).squeeze(0)  # (1,H,W,3) -> (H,W,3)
    result_bgr = cv2.cvtColor(out_im, cv2.COLOR_RGB2BGR)
    return result_bgr


def process_folder(input_folder, output_folder, model):
    os.makedirs(output_folder, exist_ok=True)

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]

    if not images:
        print(f"No image found in {input_folder}")
        return

    print(f"\nProcessing {len(images)} images from: {input_folder}")

    for i, filename in enumerate(images):
        input_path  = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        print(f"  [{i+1}/{len(images)}] {filename}")

        img = cv2.imread(input_path)
        if img is None:
            continue

        img = apply_waternet(img, model)

        img = laplacian_sharpen(img)

        img = apply_CLAHE(img)

        cv2.imwrite(output_path, img)

    print(f"  Saved in: {output_folder}")


if __name__ == "__main__":

    img_path_l = os.path.join(os.getcwd(), 'Input/left')
    img_path_r = os.path.join(os.getcwd(), 'Input/right')
    save_path_l = os.path.join(os.getcwd(), 'Output/left')
    save_path_r = os.path.join(os.getcwd(), 'Output/right')

    os.makedirs(save_path_l, exist_ok=True)
    os.makedirs(save_path_r, exist_ok=True)

    model = load_waternet()

    process_folder(img_path_l, save_path_l, model)
    process_folder(img_path_r, save_path_r, model)

    print("\nPipeline completed!")
