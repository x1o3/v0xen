import librosa
import re
from PIL import Image
import numpy as np
import hashlib

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

input_tuples = [
        (10, 4, 5), (9, 5, 1), (12, 4, 5), (10, 6, 1), (11, 7, 1), (9, 8, 1), (15, 4, 7), (14, 6, 1),
        (9, 4, 6), (13, 4, 5), (12, 6, 1), (12, 2, 7), (11, 5, 1), (15, 5, 1), (15, 4, 5), (14, 5, 1)
    ]

mapping_b = generate_mapping(input_tuples,key="s9On31havc13")

mapping = {
    '41': '1111', '42': '1110', '43': '1101', '44': '1100', '45': '1011', 
    '46': '1010', '47': '1001', '30': '1000', '31': '0111', '32': '0110', 
    '33': '0101', '69': '0100', '23': '0011', '2d': '0010'
}

replacements = {
    9: 9, 24: 25, 41: 41, 58: 57, 73: 73, 87: 89, 104: 105, 123: 121, 139: 137,
    156: 153, 165: 169, 185: 185, 196: 201, 220: 217, 233: 233, 247: 249
}

def custom_decode(encoded_str):
    result = []
    segments = encoded_str.split()
    for segment in segments:
        if segment in mapping.values():
            for k, v in mapping.items():
                if v == segment:
                    result.append(k)
                    break
    return ''.join(result)

def extract_msb_from_stego(stego_path):
    stego_img = Image.open(stego_path).convert("L")
    extracted_msb_array = np.zeros((stego_img.height, stego_img.width), dtype=np.uint8)
    for y in range(stego_img.height):
        for x in range(stego_img.width):
            pixel = stego_img.getpixel((x, y))
            extracted_msb_array[y, x] = pixel & 0b00000011
    return extracted_msb_array

def extract_lsb(image_path):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    lsb_array = img_array & 0b00000001
    return lsb_array

def decodeBits(extracted_msb_array):
    for row in extracted_msb_array:
        for i in range(1, len(row), 2):
            row[i] ^= 0b11
    return extracted_msb_array

def ber(original_image, received_image):
    original_array = np.array(original_image, dtype=np.uint8).flatten()
    received_array = np.array(received_image, dtype=np.uint8).flatten()
    original_bits = np.unpackbits(original_array.astype(np.uint8))
    received_bits = np.unpackbits(received_array.astype(np.uint8))
    bit_errors = np.sum(original_bits != received_bits)
    total_bits = original_bits.size
    bit_error_rate = bit_errors / total_bits
    return bit_error_rate

def ncc(original_image, extracted_image):
    original = np.array(original_image.convert("L"), dtype=np.float64)
    extracted = np.array(extracted_image.convert("L"), dtype=np.float64)
    numerator = np.sum((original - original.mean()) * (extracted - extracted.mean()))
    denominator = np.sqrt(np.sum((original - original.mean())**2) * np.sum((extracted - extracted.mean())**2))
    return numerator / denominator if denominator != 0 else 0

def array_conversion(extracted_msb_array):
    height, width = extracted_msb_array.shape
    new_width = width // 2
    extracted_array = np.zeros((height, new_width), dtype=np.uint8)
    for y in range(height):
        combined_row = []
        for x in range(0, width, 2):
            high_bits = extracted_msb_array[y, x] << 2
            low_bits = extracted_msb_array[y, x + 1]
            combined_value = high_bits | low_bits
            combined_row.append(combined_value)
        extracted_array[y, :len(combined_row)] = combined_row
    return extracted_array

def reverse_to_original(extracted_array, mapping_b):
    encoded_array = []
    for row in extracted_array:
        original_row = []
        for identifier in row:
            for key, value in mapping_b.items():
                if identifier == value:
                    original_row.extend(key) 
                    break
        encoded_array.append(original_row)
    return np.array(encoded_array)

def crop_image(input_path):
    img = Image.open(input_path)
    new_width = img.width // 2
    new_height = img.height // 2
    left = 0
    top = 0
    right = new_width
    bottom = new_height
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img

def decode_secret(encoded_array, cover_width, cover_height):
    rows, cols = encoded_array.shape
    decoded_array = []
    for i in range(rows):
        row = []
        for j in range(0, cols - 2, 3):
            if j + 2 < cols:
                group = encoded_array[i, j:j+3]
                binary_group = ' '.join(format(num, '04b') for num in group)
                row.append(binary_group)
        if row:
            decoded_array.append(row)
    
    hex_array = [
        [custom_decode(element) for element in row if element.strip()]
        for row in decoded_array
    ]
    
    ascii_array = []
    for row in hex_array:
        ascii_row = []
        for hex_value in row:
            if hex_value:
                try:
                    hex_pairs = [hex_value[i:i+2] for i in range(0, len(hex_value), 2)]
                    hex_pairs = [hex_pair.replace('69', '266f') for hex_pair in hex_pairs]
                    ascii_str = ''.join(chr(int(hex_pair, 16)) for hex_pair in hex_pairs)
                    pattern = r'[A-G](?:[#♯b]?[-]?[0-9]*)'
                    ascii_notes = re.findall(pattern, ascii_str)
                    if ascii_notes:
                        ascii_row.append(ascii_notes)
                except ValueError:
                    continue
        if ascii_row:
            ascii_array.append(ascii_row)
    
    note_values_array = []
    for row in ascii_array:
        note_row = []
        for notes in row:
            hz_values = []
            for note in notes:
                try:
                    hz_value = librosa.note_to_hz(note)
                    hz_values.append(round(hz_value))
                except librosa.ParameterError:
                    continue
            if hz_values:
                note_row.append(hz_values)
        if note_row:
            note_values_array.append(note_row)
    
    final_binary_array = []
    for row in note_values_array:
        binary_row = []
        for hz_values in row:
            hz_values = [replacements.get(value, value) for value in hz_values]
            binary_values = ' '.join(format(hz, '08b') for hz in hz_values)
            binary_row.append(binary_values)
        if binary_row:
            final_binary_array.append(binary_row)

    sec_img = Image.new("L", (cover_width, cover_height))
    for y in range(cover_height//2):
        for x in range(int(cover_width//2)):
            if y < len(final_binary_array) and x < len(final_binary_array[y]):
                try:
                    pixel_value = int(final_binary_array[y][x], 2)
                    sec_img.putpixel((x, y), pixel_value)
                except ValueError:
                    sec_img.putpixel((x, y), 0)
    return sec_img

def main(secret_path=None,stego_path=None,final_path=None,verbose=True):
    i = 665
    j = 2074
    if secret_path == None: secret_path = f'./Images/images/secret/resized_images/{j}.jpg'
    if stego_path == None: stego_path = f'./Images/images/stego/{i}_{j}.png'
    if final_path == None: final_path = f'./Images/images/extracted/cropped/final_{i}_{j}.png'
    output_image_path = f'./Images/images/extracted/extracted_{i}_{j}.png'
    secret_image = Image.open(secret_path)
    ex_sec = crop_image(output_image_path)
    ex_sec.save(final_path)
    final_image = Image.open(final_path)
    extracted_msb_array = extract_msb_from_stego(stego_path)
    decoded_array = decodeBits(extracted_msb_array)
    extracted_2d_array = array_conversion(decoded_array)
    final_array = reverse_to_original(extracted_2d_array,mapping_b)
    sec_img = decode_secret(final_array, *Image.open(stego_path).size )
    sec_img.save(output_image_path)
    if verbose == True : 
        print("BER: ", ber(secret_image,final_image))
        print("NCC: ", ncc(secret_image, final_image))
    return final_path

if __name__ == "__main__":
    main()