import librosa
from PIL import Image
import numpy as np
import math
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import hashlib
import time
import lpips
import torchvision.transforms as transforms

loss_fn = lpips.LPIPS(net='alex')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return mse, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def ssim(img1, img2):
    ssim_value = structural_similarity(img1, img2, data_range=255)
    return ssim_value

def ber(original_image, received_image):
    original_array = np.array(original_image, dtype=np.uint8).flatten()
    received_array = np.array(received_image, dtype=np.uint8).flatten()
    original_bits = np.unpackbits(original_array.astype(np.uint8))
    received_bits = np.unpackbits(received_array.astype(np.uint8))
    bit_errors = np.sum(original_bits != received_bits)
    total_bits = original_bits.size
    bit_error_rate = bit_errors / total_bits
    return bit_error_rate

def resize_secret_image(secret_path, cover_width, resized_image_path):
    new_width = cover_width // 2
    secret_img = Image.open(secret_path).convert("L")
    secret_aspect_ratio = secret_img.height / secret_img.width
    new_height = int(new_width * secret_aspect_ratio)
    resized_secret_img = secret_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    resized_secret_img = resized_secret_img.convert("L")
    resized_secret_img.save(resized_image_path, format="JPEG")
    return resized_secret_img

def extract_top_4_msbs(image_path):
    img = Image.open(image_path).convert("L")
    width, height = img.size
    msb_array = np.zeros((height, width), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            pixel_value = img.getpixel((x, y))
            top_4_msbs = (pixel_value >> 4) & 0b1111 
            msb_array[y, x] = top_4_msbs
    
    return msb_array

def encoding(msb_array):
    mapping = {
        '41': '1111', '42': '1110', '43': '1101', '44': '1100',
        '45': '1011', '46': '1010', '47': '1001', '30': '1000',
        '31': '0111', '32': '0110', '33': '0101', '69': '0100',
        '23': '0011', '2d': '0010'
    }
    
    def process_msb(msb):
        msb_extended = msb + '1001'
        hz = int(msb_extended, 2)
        note = librosa.hz_to_note(hz)
        hex_str = ''.join([format(ord(char), '02x') for char in note])
        hex_str = hex_str.replace('266f', '69')
        result = []
        for i in range(0, len(hex_str), 2):
            hex_pair = hex_str[i:i+2]
            for k, v in mapping.items():
                if hex_pair == k:
                    result.append(v)
                    break
        while len(result) < 3:
            result.append('0001')
        return result

    height, width = msb_array.shape
    encoded_array = np.zeros((height, width * 3), dtype=np.uint8)
    
    for y in range(height):
        for x in range(width):
            msb_binary = format(msb_array[y, x], '04b')
            encoded_values = process_msb(msb_binary)
            encoded_array[y, x * 3:x * 3 + 3] = [int(b, 2) for b in encoded_values]
    
    return encoded_array

def generate_mapping(input_tuples, key):
    mapping = {}
    used_values = set()
    for input_tuple in sorted(input_tuples):
        hash_input = (key + str(input_tuple)).encode()
        hash_value = int(hashlib.sha256(hash_input).hexdigest(), 16)
        mapped_value = hash_value % 16
        while mapped_value in used_values:
            mapped_value = (mapped_value + 1) % 16
        used_values.add(mapped_value)
        mapping[input_tuple] = mapped_value
    return mapping

def process_array(encoded_array):
    input_tuples = [
        (10, 4, 5), (9, 5, 1), (12, 4, 5), (10, 6, 1), (11, 7, 1), (9, 8, 1), (15, 4, 7), (14, 6, 1),
        (9, 4, 6), (13, 4, 5), (12, 6, 1), (12, 2, 7), (11, 5, 1), (15, 5, 1), (15, 4, 5), (14, 5, 1)
    ]

    mapping = generate_mapping(input_tuples,key="s9On31havc13")
    height, width = encoded_array.shape
    new_width = (width // 3) + 1
    mapped_array = np.zeros((height, new_width), dtype=np.uint8)
    
    for row_idx in range(height):
        row = encoded_array[row_idx]
        row_mapping = []
        
        for i in range(0, len(row), 3):
            group = tuple(row[i:i + 3])
            if group in mapping:
                row_mapping.append(mapping[group])
        mapped_array[row_idx, :len(row_mapping)] = row_mapping
    
    final_width = mapped_array.shape[1] * 2
    final_array = np.zeros((height, final_width), dtype=np.uint8)
    
    for row_idx in range(height):
        expanded_row = []
        for num in mapped_array[row_idx]:
            high_bits = (num >> 2) & 0b11
            low_bits = num & 0b11
            expanded_row.extend([high_bits, low_bits])
        
        final_array[row_idx, :len(expanded_row)] = expanded_row
    
    return final_array

def encodeBits(final_array):
    for row in final_array:
        for i in range(1, len(row), 2):
            row[i] ^= 0b11
    return final_array

def embedding(encoded_array, cover_image_path, output_path):
    start_time = time.time()
    cover_img = Image.open(cover_image_path).convert("L")
    cover_width, cover_height = cover_img.size
    stego_img = Image.new("L", (cover_width, cover_height))

    encoded_height, encoded_width = encoded_array.shape

    for y in range(cover_height):
        for x in range(cover_width):
            cover_pixel = cover_img.getpixel((x, y))

            if y < encoded_height and x < encoded_width:
                encoded_value = encoded_array[y, x]
                modified_cover_pixel = cover_pixel & 0b11111100
                stego_pixel = modified_cover_pixel | (encoded_value & 0b00000011)
                stego_img.putpixel((x, y), int(stego_pixel))
            else:
                stego_img.putpixel((x, y), cover_pixel)
    end_time = time.time()
    total_time = end_time - start_time 
    stego_img.save(output_path)
    return stego_img , total_time

def compare_histograms(cover_image_path, x, stego_path,i,j):
    cover_img = Image.open(cover_image_path).convert("L")
    stego_img = Image.open(stego_path).convert("L")
    plt.figure(figsize=(15, 10))
    cover_hist = np.histogram(np.array(cover_img).flatten(), bins=256, range=(0, 256))
    stego_hist = np.histogram(np.array(stego_img).flatten(), bins=256, range=(0, 256))
    plt.plot(cover_hist[0], label="Cover Image")
    plt.plot(stego_hist[0], label="Stego Image")
    plt.ylim(0, 1000)
    plt.legend()
    plt.savefig(f"{x}/histogram_attacks/{i}_{j}_25.png")
    plt.show()
    plt.close()
    
def save_comparison_image(cover_image_path, stego_path, x, i,j, mse, psnr_value, ssim_value):
    cover_img = Image.open(cover_image_path).convert("L")
    stego_img = Image.open(stego_path).convert("L")
    width, height = cover_img.size
    dpi = 100
    fig_width = (width * 2 + 50) / dpi
    fig_height = (height + 50) / dpi
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    ax1 = plt.subplot(1, 2, 1)
    plt.imshow(cover_img, cmap='gray', interpolation='none')
    plt.title('Cover Image')
    ax1.set_axis_off()
    ax2 = plt.subplot(1, 2, 2)
    plt.imshow(stego_img, cmap='gray', interpolation='none')
    plt.title('Stego Image')
    ax2.set_axis_off()
    
    metrics_text = (
        f'MSE: {mse:.6f} | '
        f'PSNR: {psnr_value:.6f} dB | '
        f'SSIM: {ssim_value:.8f}'
    )
    plt.figtext(0.5, 0.02, metrics_text,
                fontsize=8,
                ha='center',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0.1, wspace=0.05)
    plt.savefig(f"{x}/comparison/{i}_{j}_comparison.png", dpi=dpi, bbox_inches='tight')
    plt.close()

def main():
    i = 665
    j = 2074
    cover_image_path = (f'./Images/images/cover/{i}.jpg')
    secret_path = (f'./Images/images/secret/{j}.jpg')
    resized_image_path = (f'./Images/images/secret/resized_images/{j}.jpg')
    stego_path = (f'./Images/images/stego/{i}_{j}.png')
    x = "./Images/"
    cover_img = Image.open(cover_image_path).convert("L")
    cover_width, cover_height = cover_img.size

    resize_secret_image(secret_path, cover_width, resized_image_path)
    msb_array = extract_top_4_msbs(resized_image_path)
    encoded_array = encoding(msb_array)
    final_array = process_array(encoded_array)
    array = encodeBits(final_array)
    stego_image , total_time = embedding(array, cover_image_path, stego_path)

    original_cover_array = np.array(Image.open(cover_image_path).convert("L"))
    stego_array = np.array(stego_image)
    compare_histograms(cover_image_path,x,stego_path,i,j)

    mse, psnr_value = psnr(original_cover_array, stego_array)
    ssim_value = ssim(original_cover_array, stego_array)
    cover_tensor = transform(cover_img).unsqueeze(0) * 2 - 1
    stego_tensor = transform(stego_image).unsqueeze(0) * 2 - 1
    dist = loss_fn(cover_tensor, stego_tensor)

    save_comparison_image(cover_image_path, stego_path, x,i,j, mse, psnr_value, ssim_value)
    
    print(f'MSE: {mse}')
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_value}")
    print(f"Total Time : {total_time: .6f}")
    print("Bit-Error rate:", ber(cover_img,stego_image))
    print("LPIPS distance:", dist.item())

if __name__ == "__main__":
    main() 