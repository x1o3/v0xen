import os
import cv2
import math
import numpy as np
from PIL import Image
import torch
import lpips
import tempfile
import pandas as pd
from skimage.metrics import structural_similarity
from extract import ncc, ber
from extract import main as extraction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fn = lpips.LPIPS(net='alex').to(device)

def psnr_and_mse(imgA, imgB):
    mse = float(np.mean((imgA.astype(np.float64) - imgB.astype(np.float64)) ** 2))
    if mse == 0:
        return mse, float('inf')
    PIXEL_MAX = 255.0
    psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return mse, psnr

def ssim_value(imgA, imgB):
    return float(structural_similarity(imgA, imgB, data_range=255))

def lpips_value(pathA, pathB):
    t = lambda im: (torch.from_numpy(np.array(im).transpose(2,0,1)).float()/127.5 - 1.0).unsqueeze(0)
    imA = Image.open(pathA).convert("RGB").resize((256,256), Image.Resampling.BILINEAR)
    imB = Image.open(pathB).convert("RGB").resize((256,256), Image.Resampling.BILINEAR)
    tA, tB = t(imA).to(device), t(imB).to(device)
    with torch.no_grad():
        d = loss_fn(tA, tB)
    return float(d.cpu().item())

def add_jpeg(img_path, out_path, q):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, q])
    attacked = cv2.imread(out_path, cv2.IMREAD_COLOR)
    if attacked is None:
        raise ValueError(f"Could not read recompressed image at {out_path}")
    return attacked

def add_gaussian_noise(img_path, out_path, sigma=5.0):
    img = cv2.imread(img_path).astype(np.float32)
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, noisy)
    return out_path

def median_filter(img_path, out_path, ksize=3):
    img = cv2.imread(img_path)
    filtered = cv2.medianBlur(img, ksize)
    cv2.imwrite(out_path, filtered)
    return out_path

def gaussian_blur(img_path, out_path, ksize=5):
    img = cv2.imread(img_path)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), 0)
    cv2.imwrite(out_path, blurred)
    return out_path

def resize_image(img_path, out_path, scale=0.9):
    img = Image.open(img_path)
    w, h = img.size
    new = img.resize((int(w*scale), int(h*scale)), Image.Resampling.BILINEAR)
    new = new.resize((w, h), Image.Resampling.BILINEAR)
    new.save(out_path)
    return out_path

def rotate_image(img_path, out_path, angle=2.0):
    img = Image.open(img_path)
    rotated = img.rotate(angle, resample=Image.Resampling.BILINEAR)
    rotated.save(out_path)
    return out_path

def crop_top_left(img_path, out_path, frac=0.9):
    img = Image.open(img_path)
    w, h = img.size
    cw, ch = int(w*frac), int(h*frac)
    cropped = img.crop((0,0,cw,ch))
    canvas = Image.new(img.mode, (w,h))
    canvas.paste(cropped, (0,0))
    canvas.save(out_path)
    return out_path

def evaluate(stego_path, secret_path, attack_fn, attack_kwargs, i, j):
    tmpdir = tempfile.mkdtemp()
    attacked_path = os.path.join(tmpdir, "attacked.jpg")
    attack_fn(stego_path, attacked_path, **attack_kwargs)
    secret_image = Image.open(secret_path)
    stego = np.array(Image.open(stego_path).convert("L"))
    attacked = np.array(Image.open(attacked_path).convert("L"))
    mse, psnr = psnr_and_mse(stego, attacked)
    ssim = ssim_value(stego, attacked)
    lp = lpips_value(stego_path, attacked_path)
    attack_name = attack_fn.__name__
    param_str = "_".join(f"{k}{v}" for k, v in attack_kwargs.items())
    extracted_dir = "./Images/images/attacks"
    os.makedirs(extracted_dir, exist_ok=True)
    extracted_path = os.path.join(
        extracted_dir,
        f"{attack_name}_{i}_{j}_{param_str}.png"
    )
    extraction(
        secret_path=secret_path,
        stego_path=attacked_path,
        final_path=extracted_path,
        verbose = False
    )
    extracted_image = Image.open(extracted_path).convert("L")
    stego_image = Image.open(stego_path).convert("L")
    attacked_image = Image.open(attacked_path).convert("L")
    ber_sec = ber(secret_image, extracted_image)
    ncc_sec = ncc(secret_image, extracted_image)
    ber_steg = ber(stego_image, attacked_image)
    ncc_steg = ncc(stego_image, attacked_image)

    return {
        "attack": attack_name,
        "params": attack_kwargs,
        "MSE": mse,
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS": lp,
        "BER_SEC": ber_sec,
        "NCC_SEC": ncc_sec,
        "BER_STEG": ber_steg,
        "NCC_STEG": ncc_steg,
    }

if __name__ == "__main__":
    i = 654
    j = 2055
    cover = f"./Images/images/cover/{i}.jpg"
    stego = f"./Images/images/stego/{i}_{j}.png"
    secret = f'./Images/images/secret/resized_images/{j}.jpg'
    final = f'./Images/images/extracted/cropped/final_{i}_{j}.png'

    attacks = [
        (add_jpeg, {"q":90}),
        (add_jpeg, {"q":20}),
        (add_jpeg, {"q":30}),
        (add_gaussian_noise, {"sigma":1.0}),
        (add_gaussian_noise, {"sigma":3.0}),
        (add_gaussian_noise, {"sigma":5.0}),
        (median_filter, {"ksize":1}),
        (median_filter, {"ksize":3}),
        (median_filter, {"ksize":5}),
        (gaussian_blur, {"ksize":1}),
        (gaussian_blur, {"ksize":3}),
        (gaussian_blur, {"ksize":5}),
        (resize_image, {"scale":0.5}),
        (rotate_image, {"angle":0.5})
    ]
    
    results= []

    for fn, kw in attacks:
        result = evaluate(stego, secret, fn, kw, i, j)
        results.append(result)
    
    df = pd.DataFrame(results)
    df = df[["attack", "params", "MSE", "PSNR", "SSIM", "LPIPS", "BER_SEC", "NCC_SEC", "BER_STEG", "NCC_STEG"]]
    print(df.to_markdown(index=False))