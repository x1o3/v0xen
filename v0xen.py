from skimage.metrics import structural_similarity
import sys
import os
import argparse
import time
import numpy as np
import hashlib
import struct
import zlib
import math
from typing import Union
from PIL import Image
from scipy.ndimage import sobel
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
SALT_LEN = 16
NONCE_LEN = 12
KEY_LEN = 32
FEATURE_HASH_LEN = 32

def derive_key(master_secret: bytes, salt: bytes, info: bytes=b'stego-v2-payload') -> bytes:
    hkdf = HKDF(algorithm=hashes.SHA256(), length=KEY_LEN, salt=salt, info=info)
    return hkdf.derive(master_secret)

def encrypt_payload(plaintext: bytes, master_secret: bytes) -> dict:
    salt = os.urandom(SALT_LEN)
    nonce = os.urandom(NONCE_LEN)
    key = derive_key(master_secret, salt)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data=None)
    return {'salt': salt, 'nonce': nonce, 'ciphertext': ciphertext}

def decrypt_payload(ciphertext: bytes, salt: bytes, nonce: bytes, master_secret: bytes) -> bytes:
    key = derive_key(master_secret, salt)
    aesgcm = AESGCM(key)
    return aesgcm.decrypt(nonce, ciphertext, associated_data=None)
_V2_FIXED_FMT = '>B16s12sIHH32sBHHI'
_V2_FIXED_LEN = struct.calcsize(_V2_FIXED_FMT)

def pack_header_v2(salt: bytes, nonce: bytes, payload_len: int, width: int, height: int, feature_hash: bytes, block_size: int, cover_block_rows: int, cover_block_cols: int, plan_blob: bytes) -> bytes:
    assert len(salt) == SALT_LEN and len(nonce) == NONCE_LEN
    assert len(feature_hash) == FEATURE_HASH_LEN
    assert 1 <= block_size <= 255
    fixed = struct.pack(_V2_FIXED_FMT, 3, salt, nonce, payload_len, width, height, feature_hash, block_size, cover_block_rows, cover_block_cols, len(plan_blob))
    return fixed + plan_blob

def unpack_header(header_bytes: bytes) -> dict:
    if len(header_bytes) < _V2_FIXED_LEN:
        raise ValueError(f'V2 header too short: {len(header_bytes)} < {_V2_FIXED_LEN}')
    version = header_bytes[0]
    if version != 3:
        raise ValueError(f'Unsupported stego header version: {version} (expected 3 - adaptive-bpp plan format). Images embedded with the old fixed-2bpp build must be re-embedded.')
    _, salt, nonce, payload_len, width, height, feature_hash, block_size, cover_block_rows, cover_block_cols, plan_blob_len = struct.unpack(_V2_FIXED_FMT, header_bytes[:_V2_FIXED_LEN])
    plan_blob = header_bytes[_V2_FIXED_LEN:_V2_FIXED_LEN + plan_blob_len]
    if len(plan_blob) < plan_blob_len:
        raise ValueError(f'V2 header plan blob truncated: got {len(plan_blob)}, expected {plan_blob_len}')
    return {'version': 3, 'salt': salt, 'nonce': nonce, 'payload_len': payload_len, 'width': width, 'height': height, 'feature_hash': feature_hash, 'block_size': block_size, 'cover_block_rows': cover_block_rows, 'cover_block_cols': cover_block_cols, 'plan_blob': plan_blob}

V2_FIXED_LEN = _V2_FIXED_LEN
BITS_PER_PIXEL = 2                                                        
MODULUS = 1 << BITS_PER_PIXEL                                                                            
PAYLOAD_BPP_LOW = 1
PAYLOAD_BPP_HIGH = 2
PAYLOAD_BPP_SKIP = 0
DEFAULT_BPP_TIERS = [(0.80, PAYLOAD_BPP_HIGH), (0.95, PAYLOAD_BPP_LOW), (1.0, PAYLOAD_BPP_SKIP)]

def parse_bpp_tiers(spec: str) -> list:
    """Parses '0.80:2,0.95:1,1.0:0' into [(0.80,2),(0.95,1),(1.0,0)]."""
    tiers = []
    for part in spec.split(','):
        frac_s, bpp_s = part.split(':')
        tiers.append((float(frac_s), int(bpp_s)))
    tiers.sort(key=lambda t: t[0])
    return tiers

def _seed_from_key(key: bytes, tag: bytes) -> int:
    digest = hashlib.sha256(tag + key).digest()
    return int.from_bytes(digest[:8], 'big')

def pixel_order(num_pixels: int, count: int, key: bytes, tag: bytes) -> np.ndarray:
    seed = _seed_from_key(key, tag)
    rng = np.random.default_rng(seed)
    if count > num_pixels:
        raise ValueError(f'Requested {count} pixel slots but image only has {num_pixels}.')
    return rng.permutation(num_pixels)[:count]

def lsb_match_encode(pixel_values: np.ndarray, target_bits: np.ndarray, modulus: Union[int, np.ndarray]=MODULUS) -> np.ndarray:
    pixel_values = pixel_values.astype(np.int16)
    target_bits = target_bits.astype(np.int16)
    modulus = np.asarray(modulus, dtype=np.int16)
    current = pixel_values % modulus
    delta_up = (target_bits - current) % modulus
    delta_down = delta_up - modulus
    new_up = pixel_values + delta_up
    new_down = pixel_values + delta_down
    up_valid = (new_up >= 0) & (new_up <= 255)
    down_valid = (new_down >= 0) & (new_down <= 255)
    prefer_down = np.abs(delta_down) < np.abs(delta_up)
    result = pixel_values.copy()
    use_down = prefer_down & down_valid
    result = np.where(use_down, new_down, result)
    fallback_up = prefer_down & ~down_valid & up_valid
    result = np.where(fallback_up, new_up, result)
    use_up = ~prefer_down & up_valid
    result = np.where(use_up, new_up, result)
    fallback_down = ~prefer_down & ~up_valid & down_valid
    result = np.where(fallback_down, new_down, result)
    both_invalid = ~up_valid & ~down_valid
    if np.any(both_invalid):
        clipped = np.clip(pixel_values + delta_up, 0, 255)
        result = np.where(both_invalid, clipped, result)
    return np.clip(result, 0, 255).astype(np.uint8)

def lsb_match_decode(pixel_values: np.ndarray, modulus: Union[int, np.ndarray]=MODULUS) -> np.ndarray:
    modulus = np.asarray(modulus, dtype=np.int16)
    return (pixel_values.astype(np.int16) % modulus).astype(np.uint8)

def bytes_to_2bit_chunks(data: bytes) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    bits = bits.reshape(-1, BITS_PER_PIXEL)
    weights = np.array([2, 1])
    return bits.dot(weights).astype(np.uint8)

def chunks_to_bytes(chunks: np.ndarray, num_bytes: int) -> bytes:
    needed_chunks = num_bytes * (8 // BITS_PER_PIXEL)
    chunks = chunks[:needed_chunks]
    bits = np.zeros((len(chunks), BITS_PER_PIXEL), dtype=np.uint8)
    bits[:, 0] = chunks >> 1 & 1
    bits[:, 1] = chunks & 1
    flat_bits = bits.reshape(-1)
    byte_arr = np.packbits(flat_bits)
    return byte_arr[:num_bytes].tobytes()

def bytes_to_bitarray(data: bytes) -> np.ndarray:
    """Flat MSB-first bit array, one uint8 (0/1) per bit."""
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))

def pack_bits_to_symbols(bits: np.ndarray, bpp_per_symbol: np.ndarray) -> np.ndarray:
    n = len(bpp_per_symbol)
    symbols = np.zeros(n, dtype=np.int16)
    idx = 0
    total_bits = len(bits)
    for i in range(n):
        bpp = int(bpp_per_symbol[i])
        val = 0
        for _ in range(bpp):
            bit = int(bits[idx]) if idx < total_bits else 0
            val = (val << 1) | bit
            idx += 1
        symbols[i] = val
    return symbols

def unpack_symbols_to_bits(symbols: np.ndarray, bpp_per_symbol: np.ndarray, total_bits_needed: int) -> np.ndarray:
    out = np.zeros(total_bits_needed, dtype=np.uint8)
    idx = 0
    for i in range(len(symbols)):
        if idx >= total_bits_needed:
            break
        bpp = int(bpp_per_symbol[i])
        val = int(symbols[i])
        for b in range(bpp - 1, -1, -1):
            if idx >= total_bits_needed:
                break
            out[idx] = (val >> b) & 1
            idx += 1
    return out

def bitarray_to_bytes(bits: np.ndarray, num_bytes: int) -> bytes:
    needed = num_bytes * 8
    if len(bits) < needed:
        bits = np.pad(bits, (0, needed - len(bits)), 'constant')
    return np.packbits(bits[:needed]).tobytes()
DEFAULT_BLOCK_SIZE = 8
HEADER_TAG = b'header-v2'
PAYLOAD_TAG = b'payload-v2'

def image_to_payload(image_or_path: Union[str, Image.Image]) -> dict:
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path).convert('L')
    else:
        img = image_or_path.convert('L')
    width, height = img.size
    raw = img.tobytes()
    compressed = zlib.compress(raw, level=9)
    return {'payload': compressed, 'width': width, 'height': height, 'raw_len': len(raw)}

def payload_to_image(payload: bytes, width: int, height: int) -> Image.Image:
    raw = zlib.decompress(payload)
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
    return Image.fromarray(arr)
_F_MIN_HZ = 65.41
_OCTAVE_SPAN = 5
_SEMITONE_SPAN = _OCTAVE_SPAN * 12

def extract_block_features(image_array: np.ndarray, block_size: int=8) -> dict:
    h, w = image_array.shape[:2]
    img = image_array.astype(np.float64)
    bh = max(1, (h + block_size - 1) // block_size)
    bw = max(1, (w + block_size - 1) // block_size)
    mean_lum = np.zeros((bh, bw), dtype=np.float64)
    variance = np.zeros((bh, bw), dtype=np.float64)
    grad_energy = np.zeros((bh, bw), dtype=np.float64)
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    for bi in range(bh):
        r0 = bi * block_size
        r1 = min(r0 + block_size, h)
        for bj in range(bw):
            c0 = bj * block_size
            c1 = min(c0 + block_size, w)
            block = img[r0:r1, c0:c1]
            mean_lum[bi, bj] = block.mean()
            variance[bi, bj] = block.var()
            grad_energy[bi, bj] = grad_mag[r0:r1, c0:c1].mean()
    return {'mean_luminance': mean_lum, 'variance': variance, 'gradient_energy': grad_energy, 'block_grid_shape': (bh, bw), 'block_size': block_size}

def _normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = (arr.min(), arr.max())
    if hi - lo < 1e-12:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)

def features_to_frequency_map(block_features: dict) -> np.ndarray:
    norm_mean = _normalize(block_features['mean_luminance'])
    norm_var = _normalize(block_features['variance'])
    norm_grad = _normalize(block_features['gradient_energy'])
    combined = 0.25 * norm_mean + 0.4 * norm_var + 0.35 * norm_grad
    combined = np.clip(combined, 0.0, 1.0)
    semitone = combined * _SEMITONE_SPAN
    freq_hz = _F_MIN_HZ * 2.0 ** (semitone / 12.0)
    return freq_hz

def frequency_to_energy_map(frequency_map: np.ndarray) -> np.ndarray:
    return _normalize(frequency_map)

def compute_musical_energy_map(image_array: np.ndarray, block_size: int=8) -> dict:
    feats = extract_block_features(image_array, block_size)
    freq_map = features_to_frequency_map(feats)
    energy_map = frequency_to_energy_map(freq_map)
    return {'frequency_map': freq_map, 'energy_map': energy_map, 'block_features': feats, 'block_size': block_size, 'block_grid_shape': feats['block_grid_shape']}

def compute_feature_hash(energy_map: np.ndarray) -> bytes:
    quantized = np.clip(energy_map * 255.0, 0, 255).astype(np.uint8)
    return hashlib.sha256(quantized.tobytes()).digest()
DEFAULT_WEIGHTS = {'texture': 0.35, 'entropy': 0.25, 'edge': 0.2, 'musical': 0.2}

def compute_distortion_cost(cover_analysis: dict, musical_energy_map: np.ndarray, weights: dict | None=None) -> np.ndarray:
    w = weights if weights is not None else DEFAULT_WEIGHTS
    texture = cover_analysis['texture_map']
    entropy = cover_analysis['entropy_map']
    edge = cover_analysis['edge_map']
    if musical_energy_map.shape != texture.shape:
        from scipy.ndimage import zoom
        target_shape = texture.shape
        zoom_factors = (target_shape[0] / musical_energy_map.shape[0], target_shape[1] / musical_energy_map.shape[1])
        musical = zoom(musical_energy_map, zoom_factors, order=1)
        musical = np.clip(musical, 0.0, 1.0)
    else:
        musical = musical_energy_map
    suitability = w['texture'] * texture + w['entropy'] * entropy + w['edge'] * edge + w['musical'] * musical
    cost = 1.0 - np.clip(suitability, 0.0, 1.0)
    return cost

def compute_actual_block_pixels(cover_shape: tuple, block_size: int, header_indices: np.ndarray | None=None) -> np.ndarray:
    h, w = cover_shape
    bh = max(1, (h + block_size - 1) // block_size)
    bw = max(1, (w + block_size - 1) // block_size)
    header_set = set(header_indices.tolist()) if header_indices is not None else set()
    actual = np.zeros((bh, bw), dtype=np.int64)
    for bi in range(bh):
        r0 = bi * block_size
        r1 = min(r0 + block_size, h)
        for bj in range(bw):
            c0 = bj * block_size
            c1 = min(c0 + block_size, w)
            block_pixels = (r1 - r0) * (c1 - c0)
            if header_set:
                header_in_block = 0
                for r in range(r0, r1):
                    for c in range(c0, c1):
                        if r * w + c in header_set:
                            header_in_block += 1
                block_pixels -= header_in_block
            actual[bi, bj] = max(0, block_pixels)
    return actual

def compute_block_capacity(distortion_cost: np.ndarray, total_payload_chunks: int, block_size: int=8, min_cost_threshold: float=0.92, max_capacity_per_block: np.ndarray | None=None) -> np.ndarray:
    if max_capacity_per_block is not None:
        max_cap = max_capacity_per_block.astype(np.int64)
    else:
        max_cap = np.full(distortion_cost.shape, block_size * block_size, dtype=np.int64)
    suitability = 1.0 - distortion_cost
    suitability[distortion_cost > min_cost_threshold] = 0.0
    total_suitability = suitability.sum()
    if total_suitability < 1e-12:
        num_blocks = distortion_cost.size
        base = total_payload_chunks // num_blocks
        remainder = total_payload_chunks % num_blocks
        capacity = np.full(distortion_cost.shape, base, dtype=np.int64)
        flat_idx = np.argsort(distortion_cost.flatten())
        for i in range(remainder):
            bi, bj = np.unravel_index(flat_idx[i], distortion_cost.shape)
            capacity[bi, bj] += 1
        capacity = np.minimum(capacity, max_cap)
    else:
        raw_capacity = suitability / total_suitability * total_payload_chunks
        capacity = np.floor(raw_capacity).astype(np.int64)
        capacity = np.minimum(capacity, max_cap)
        shortfall = total_payload_chunks - capacity.sum()
        if shortfall > 0:
            fractional = raw_capacity - np.floor(raw_capacity)
            room = max_cap - capacity
            fractional[room <= 0] = -1.0
            flat_order = np.argsort(-fractional.flatten())
            for i in range(min(int(shortfall), len(flat_order))):
                bi, bj = np.unravel_index(flat_order[i], capacity.shape)
                if capacity.sum() >= total_payload_chunks:
                    break
                if capacity[bi, bj] < max_cap[bi, bj]:
                    capacity[bi, bj] += 1
    while capacity.sum() < total_payload_chunks:
        room = max_cap - capacity
        if room.max() <= 0:
            break
        bi, bj = np.unravel_index(np.argmax(room), capacity.shape)
        needed = total_payload_chunks - capacity.sum()
        capacity[bi, bj] += min(needed, room[bi, bj])
    return capacity

def assign_block_bpp(cost_map: np.ndarray, tiers: list=None) -> np.ndarray:
    if tiers is None:
        tiers = DEFAULT_BPP_TIERS
    flat_cost = cost_map.flatten()
    n = flat_cost.size
    order = np.argsort(flat_cost)                                          
    bpp_flat = np.zeros(n, dtype=np.uint8)
    prev_cut = 0
    for cum_frac, bpp in tiers:
        cut = max(prev_cut, min(int(round(cum_frac * n)), n))
        bpp_flat[order[prev_cut:cut]] = bpp
        prev_cut = cut
    if prev_cut < n and tiers:
        bpp_flat[order[prev_cut:]] = tiers[-1][1]
    return bpp_flat.reshape(cost_map.shape)

def generate_embedding_plan(cover_array: np.ndarray, musical_energy_map: np.ndarray, total_payload_bits: int, block_size: int=8, weights: dict | None=None, header_indices: np.ndarray | None=None, bpp_tiers: list=None) -> dict:
    ca = analyze_cover(cover_array, block_size)
    cost = compute_distortion_cost(ca, musical_energy_map, weights)
    bpp_map = assign_block_bpp(cost, tiers=bpp_tiers)
    actual_pixels = compute_actual_block_pixels(cover_array.shape, block_size, header_indices)
    max_bit_capacity = actual_pixels * bpp_map
    bit_alloc = compute_block_capacity(cost, total_payload_bits, block_size, max_capacity_per_block=max_bit_capacity)                                              
    safe_bpp = np.maximum(bpp_map, 1)
    pixel_alloc = np.minimum(np.ceil(bit_alloc / safe_bpp).astype(np.int64), actual_pixels)
    priority = np.argsort(cost.flatten())
    true_max_bits = int(max_bit_capacity.sum())
    return {'block_capacities': pixel_alloc, 'block_priority_order': priority, 'distortion_cost_map': cost, 'cover_analysis': ca, 'bpp_map': bpp_map, 'bit_allocation': bit_alloc, 'actual_pixels': actual_pixels, 'true_max_bits': true_max_bits}

def plan_to_pixel_indices(embedding_plan: dict, cover_shape: tuple, key: bytes, tag: bytes, header_indices: np.ndarray, block_size: int=8) -> np.ndarray:
    h, w = cover_shape
    capacities = embedding_plan['block_capacities']
    priority = embedding_plan['block_priority_order']
    bh, bw = capacities.shape
    header_set = set(header_indices.tolist())
    seed_digest = hashlib.sha256(tag + key).digest()
    seed = int.from_bytes(seed_digest[:8], 'big')
    rng = np.random.default_rng(seed)
    all_indices = []
    for flat_idx in priority:
        bi, bj = np.unravel_index(flat_idx, (bh, bw))
        cap = int(capacities[bi, bj])
        if cap <= 0:
            continue
        r0 = bi * block_size
        r1 = min(r0 + block_size, h)
        c0 = bj * block_size
        c1 = min(c0 + block_size, w)
        rows = np.arange(r0, r1)
        cols = np.arange(c0, c1)
        rr, cc = np.meshgrid(rows, cols, indexing='ij')
        block_flat = (rr * w + cc).flatten()
        block_flat = np.array([p for p in block_flat if p not in header_set])
        if len(block_flat) == 0:
            continue
        rng.shuffle(block_flat)
        take = min(cap, len(block_flat))
        all_indices.append(block_flat[:take])
    if len(all_indices) == 0:
        return np.array([], dtype=np.int64)
    return np.concatenate(all_indices).astype(np.int64)

def serialize_plan(embedding_plan: dict, energy_map: np.ndarray) -> bytes:
    capacities = embedding_plan['block_capacities']
    priority = embedding_plan['block_priority_order']
    bpp_map = embedding_plan['bpp_map']
    cover_bh, cover_bw = capacities.shape
    energy_bh, energy_bw = energy_map.shape
    energy_q = np.clip(energy_map * 255.0, 0, 255).astype(np.uint8)
    header = struct.pack('>HHHH', cover_bh, cover_bw, energy_bh, energy_bw)
    cap_bytes = capacities.astype(np.int16).tobytes()
    pri_bytes = priority.astype(np.int32).tobytes()
    bpp_bytes = bpp_map.astype(np.uint8).tobytes()
    energy_bytes = energy_q.tobytes()
    raw = header + cap_bytes + pri_bytes + bpp_bytes + energy_bytes
    return zlib.compress(raw, level=9)

def deserialize_plan(plan_blob: bytes, cover_block_rows: int, cover_block_cols: int) -> dict:
    raw = zlib.decompress(plan_blob)
    hdr_size = struct.calcsize('>HHHH')
    cover_bh, cover_bw, energy_bh, energy_bw = struct.unpack('>HHHH', raw[:hdr_size])
    offset = hdr_size
    cap_size = cover_bh * cover_bw * 2
    capacities = np.frombuffer(raw[offset:offset + cap_size], dtype=np.int16)
    capacities = capacities.reshape(cover_bh, cover_bw).copy()
    offset += cap_size
    pri_size = cover_bh * cover_bw * 4
    priority = np.frombuffer(raw[offset:offset + pri_size], dtype=np.int32).copy()
    offset += pri_size
    bpp_size = cover_bh * cover_bw
    bpp_map = np.frombuffer(raw[offset:offset + bpp_size], dtype=np.uint8)
    bpp_map = bpp_map.reshape(cover_bh, cover_bw).copy()
    offset += bpp_size
    energy_size = energy_bh * energy_bw
    energy_q = np.frombuffer(raw[offset:offset + energy_size], dtype=np.uint8)
    energy_map = energy_q.reshape(energy_bh, energy_bw).astype(np.float64) / 255.0
    offset += energy_size
    return {'block_capacities': capacities, 'block_priority_order': priority, 'bpp_map': bpp_map, 'energy_map': energy_map}

def compute_texture_map(cover_array: np.ndarray, block_size: int=8) -> np.ndarray:
    h, w = cover_array.shape[:2]
    img = cover_array.astype(np.float64)
    bh = max(1, (h + block_size - 1) // block_size)
    bw = max(1, (w + block_size - 1) // block_size)
    texture = np.zeros((bh, bw), dtype=np.float64)
    for bi in range(bh):
        r0, r1 = (bi * block_size, min((bi + 1) * block_size, h))
        for bj in range(bw):
            c0, c1 = (bj * block_size, min((bj + 1) * block_size, w))
            texture[bi, bj] = img[r0:r1, c0:c1].var()
    return _normalize(texture)

def compute_entropy_map(cover_array: np.ndarray, block_size: int=8) -> np.ndarray:
    h, w = cover_array.shape[:2]
    bh = max(1, (h + block_size - 1) // block_size)
    bw = max(1, (w + block_size - 1) // block_size)
    entropy = np.zeros((bh, bw), dtype=np.float64)
    for bi in range(bh):
        r0, r1 = (bi * block_size, min((bi + 1) * block_size, h))
        for bj in range(bw):
            c0, c1 = (bj * block_size, min((bj + 1) * block_size, w))
            block = cover_array[r0:r1, c0:c1].flatten()
            counts = np.bincount(block, minlength=256).astype(np.float64)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy[bi, bj] = -np.sum(probs * np.log2(probs))
    return _normalize(entropy)

def compute_edge_map(cover_array: np.ndarray, block_size: int=8) -> np.ndarray:
    h, w = cover_array.shape[:2]
    img = cover_array.astype(np.float64)
    bh = max(1, (h + block_size - 1) // block_size)
    bw = max(1, (w + block_size - 1) // block_size)
    gx = sobel(img, axis=1)
    gy = sobel(img, axis=0)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    edge = np.zeros((bh, bw), dtype=np.float64)
    for bi in range(bh):
        r0, r1 = (bi * block_size, min((bi + 1) * block_size, h))
        for bj in range(bw):
            c0, c1 = (bj * block_size, min((bj + 1) * block_size, w))
            edge[bi, bj] = grad_mag[r0:r1, c0:c1].mean()
    return _normalize(edge)

def analyze_cover(cover_array: np.ndarray, block_size: int=8) -> dict:
    texture = compute_texture_map(cover_array, block_size)
    entropy = compute_entropy_map(cover_array, block_size)
    edge = compute_edge_map(cover_array, block_size)
    bh, bw = texture.shape
    return {'texture_map': texture, 'entropy_map': entropy, 'edge_map': edge, 'block_size': block_size, 'block_grid_shape': (bh, bw)}
from scipy.stats import chisquare, ks_2samp

def psnr_mse(img1: np.ndarray, img2: np.ndarray) -> dict:
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse == 0:
        return {'mse': 0.0, 'psnr_db': float('inf')}
    psnr_db = 20 * math.log10(255.0 / math.sqrt(mse))
    return {'mse': mse, 'psnr_db': psnr_db}

def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(structural_similarity(img1, img2, data_range=255))

def bit_error_rate(bytes1: bytes, bytes2: bytes) -> dict:
    n = min(len(bytes1), len(bytes2))
    a = np.frombuffer(bytes1[:n], dtype=np.uint8)
    b = np.frombuffer(bytes2[:n], dtype=np.uint8)
    bits_a = np.unpackbits(a)
    bits_b = np.unpackbits(b)
    errors = int(np.sum(bits_a != bits_b))
    return {'length_match': len(bytes1) == len(bytes2), 'compared_bytes': n, 'bit_errors': errors, 'total_bits': bits_a.size, 'ber': errors / bits_a.size if bits_a.size else 0.0}

def image_ber(img1: np.ndarray, img2: np.ndarray) -> dict:
    a = np.unpackbits(img1.flatten().astype(np.uint8))
    b = np.unpackbits(img2.flatten().astype(np.uint8))
    n = min(len(a), len(b))
    errors = int(np.sum(a[:n] != b[:n]))
    return {'bit_errors': errors, 'total_bits': n, 'ber': errors / n if n else 0.0}

def ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    a = img1.flatten().astype(np.float64)
    b = img2.flatten().astype(np.float64)
    a_mean = a - a.mean()
    b_mean = b - b.mean()
    num = np.sum(a_mean * b_mean)
    denom = np.sqrt(np.sum(a_mean ** 2) * np.sum(b_mean ** 2))
    if denom < 1e-12:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(num / denom)

def histogram_tests(cover_arr: np.ndarray, stego_arr: np.ndarray) -> dict:
    cover_hist, _ = np.histogram(cover_arr.flatten(), bins=256, range=(0, 256))
    stego_hist, _ = np.histogram(stego_arr.flatten(), bins=256, range=(0, 256))
    expected = cover_hist.astype(np.float64) + 1e-06
    observed = stego_hist.astype(np.float64) + 1e-06
    expected = expected * (observed.sum() / expected.sum())
    chi2_stat, chi2_p = chisquare(observed, expected)
    ks_stat, ks_p = ks_2samp(cover_arr.flatten(), stego_arr.flatten())
    return {'chi_square_stat': float(chi2_stat), 'chi_square_p': float(chi2_p), 'ks_stat': float(ks_stat), 'ks_p': float(ks_p), 'note': 'Low p-values indicate a statistically detectable difference between cover and stego histograms (a steganalysis red flag).'}

def true_capacity_bytes(cover_path: str, block_size: int=DEFAULT_BLOCK_SIZE, bpp_tiers: list=None, header_len: int=None) -> int:
    cover_img = Image.open(cover_path).convert('L')
    cover_arr = np.array(cover_img, dtype=np.uint8)
    cover_h, cover_w = cover_arr.shape
    ca = analyze_cover(cover_arr, block_size)
    energy_bh = max(1, (cover_h + block_size - 1) // block_size)
    energy_bw = max(1, (cover_w + block_size - 1) // block_size)
    zero_energy = np.zeros((energy_bh, energy_bw))
    cost = compute_distortion_cost(ca, zero_energy)
    bpp_map = assign_block_bpp(cost, tiers=bpp_tiers)
    actual_pixels = compute_actual_block_pixels(cover_arr.shape, block_size, None)
    true_max_bits = int((actual_pixels * bpp_map).sum())
    if header_len is None:
        est_bh = max(1, (cover_h + block_size - 1) // block_size)
        est_bw = max(1, (cover_w + block_size - 1) // block_size)
        header_len = V2_FIXED_LEN + int(6 * est_bh * est_bw * 0.4)
    return max(0, (true_max_bits - header_len * 8) // 8)

def capacity_bytes(cover_width: int, cover_height: int, header_len: int=None) -> int:
    if header_len is None:
        header_len = V2_FIXED_LEN
    total_pixels = cover_width * cover_height
    header_chunks_needed = header_len * (8 // BITS_PER_PIXEL)
    usable_pixels = total_pixels - header_chunks_needed
    return max(0, usable_pixels * BITS_PER_PIXEL // 8)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class EmbeddingPipeline:

    def __init__(self, cover_path: str, secret_path: str, master_secret: bytes, output_path: str, block_size: int=DEFAULT_BLOCK_SIZE, verbose: bool=True, bpp_tiers: list=None):
        self.cover_path = cover_path
        self.secret_path = secret_path
        self.master_secret = master_secret
        self.output_path = output_path
        self.block_size = block_size
        self.verbose = verbose
        self.bpp_tiers = bpp_tiers
        self._step = 0

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _step_header(self, title: str):
        self._step += 1
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f'  Step {self._step}: {title}')
            print(f"{'=' * 60}")

    def run(self) -> dict:
        t0 = time.time()
        self._step_header('Load Cover Image')
        if not os.path.exists(self.cover_path):
            raise FileNotFoundError(f'Cover image not found: {self.cover_path}')
        cover_img = Image.open(self.cover_path).convert('L')
        cover_w, cover_h = cover_img.size
        cover_arr = np.array(cover_img, dtype=np.uint8)
        cover_pixels = cover_arr.flatten()
        total_pixels = cover_pixels.size
        self._log(f'  Image: {self.cover_path}')
        self._log(f'  Dimensions: {cover_w} x {cover_h}')
        self._log(f'  Total pixels: {total_pixels:,}')
        self._step_header('Fit Secret Image into Cover Capacity')
        if not os.path.exists(self.secret_path):
            raise FileNotFoundError(f'Secret image not found: {self.secret_path}')
        secret_img = Image.open(self.secret_path).convert('L')
        cover_bh = max(1, (cover_h + self.block_size - 1) // self.block_size)
        cover_bw = max(1, (cover_w + self.block_size - 1) // self.block_size)
        est_plan_blob = int(6 * cover_bh * cover_bw * 0.4)
        cap = capacity_bytes(cover_w, cover_h, V2_FIXED_LEN + est_plan_blob)
        orig_secret_img = secret_img.copy()
        orig_w, orig_h = orig_secret_img.size
        
        low_scale = 0.0
        high_scale = 1.0
        resized = False
        
        while True:
            secret_arr = np.array(secret_img, dtype=np.uint8)
            secret_w, secret_h = secret_img.size
            musical = compute_musical_energy_map(secret_arr, self.block_size)
            energy_map = musical['energy_map']
            feature_hash = compute_feature_hash(energy_map)
            secret_info = image_to_payload(secret_img)
            compressed = secret_info['payload']
            needed = len(compressed) + 16
            total_payload_bits = needed * 8
            prelim_plan = generate_embedding_plan(cover_arr, energy_map, total_payload_bits=total_payload_bits, block_size=self.block_size, bpp_tiers=self.bpp_tiers)
            prelim_blob = serialize_plan(prelim_plan, energy_map)
            padding = max(128, int(len(prelim_blob) * 0.15))
            target_blob_len = len(prelim_blob) + padding
            true_header_len = V2_FIXED_LEN + target_blob_len                                            
            header_bits_reserved = true_header_len * 8
            true_cap_bits = max(0, prelim_plan['true_max_bits'] - header_bits_reserved)
            true_cap = true_cap_bits // 8                                                                                       
            if needed <= true_cap and (not resized or needed >= true_cap * 0.90):
                break                           
            if resized:
                if needed > true_cap:
                    high_scale = min(high_scale, secret_w / orig_w)
                else:
                    low_scale = max(low_scale, secret_w / orig_w)
            if needed > true_cap:        
                target_scale = math.sqrt(true_cap / needed) * 0.98
                next_scale = (secret_w / orig_w) * target_scale
            else:                                   
                next_scale = (low_scale + high_scale) / 2.0  
            new_w = max(1, int(orig_w * next_scale))
            new_h = max(1, int(orig_h * next_scale))              
            if new_w == secret_w and new_h == secret_h:
                if needed > true_cap:
                    new_w = max(1, new_w - 1)
                    new_h = max(1, new_h - 1)
                else:
                    break 
            self._log(f'  Payload ({needed:,} B) vs True Cap ({true_cap:,} B). Resizing to {new_w}x{new_h}...')
            secret_img = orig_secret_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            resized = True
            
        if resized:
            orig_name = os.path.basename(self.secret_path)
            resized_dir = os.path.join(os.path.dirname(self.secret_path), 'resized')
            os.makedirs(resized_dir, exist_ok=True)
            resized_path = os.path.join(resized_dir, orig_name)
            secret_img.save(resized_path)
            self._log(f'  Saved resized secret image to: {resized_path}')

        self._log(f'  Final Secret Image: {secret_w} x {secret_h}')
        self._log(f'  Compressed Payload: {len(compressed):,} bytes')
        self._log(f'  Feature hash: {feature_hash.hex()}')
        self._step_header('AES-256-GCM Encryption')
        env = encrypt_payload(compressed, self.master_secret)
        self._log(f"  Salt: {env['salt'].hex()}")
        self._log(f"  Nonce: {env['nonce'].hex()}")
        self._log(f'  Plaintext: {len(compressed):,} bytes')
        self._log(f"  Ciphertext: {len(env['ciphertext']):,} bytes (+16 bytes auth tag)")
        self._step_header('Cover Image Analysis')
        ca = analyze_cover(cover_arr, self.block_size)
        cover_bh, cover_bw = ca['block_grid_shape']
        self._log(f'  Block grid: {cover_bw} x {cover_bh} = {cover_bw * cover_bh} blocks')
        self._log(f"  Texture map:  mean={ca['texture_map'].mean():.3f}, std={ca['texture_map'].std():.3f}")
        self._log(f"  Entropy map:  mean={ca['entropy_map'].mean():.3f}, std={ca['entropy_map'].std():.3f}")
        self._log(f"  Edge map:     mean={ca['edge_map'].mean():.3f}, std={ca['edge_map'].std():.3f}")
        self._step_header('Adaptive Distortion Cost Computation')
        total_payload_bits = len(env['ciphertext']) * 8
        prelim_plan = generate_embedding_plan(cover_arr, energy_map, total_payload_bits=total_payload_bits, block_size=self.block_size, bpp_tiers=self.bpp_tiers)
        prelim_blob = serialize_plan(prelim_plan, energy_map)
        padding = max(128, int(len(prelim_blob) * 0.15))
        target_blob_len = len(prelim_blob) + padding
        padded_blob = prelim_blob.ljust(target_blob_len, b'\x00')
        prelim_header = pack_header_v2(salt=env['salt'], nonce=env['nonce'], payload_len=len(env['ciphertext']), width=secret_w, height=secret_h, feature_hash=feature_hash, block_size=self.block_size, cover_block_rows=cover_bh, cover_block_cols=cover_bw, plan_blob=padded_blob)
        header_idx = pixel_order(total_pixels, len(bytes_to_2bit_chunks(prelim_header)), key=b'public', tag=HEADER_TAG)
        plan = generate_embedding_plan(cover_arr, energy_map, total_payload_bits=total_payload_bits, block_size=self.block_size, header_indices=header_idx, bpp_tiers=self.bpp_tiers)
        final_blob = serialize_plan(plan, energy_map)
        if len(final_blob) > target_blob_len:
            raise RuntimeError(f'Plan blob size exceeded padding margin ({len(final_blob)} > {target_blob_len}).')
        plan_blob = final_blob.ljust(target_blob_len, b'\x00')
        header_bytes = pack_header_v2(salt=env['salt'], nonce=env['nonce'], payload_len=len(env['ciphertext']), width=secret_w, height=secret_h, feature_hash=feature_hash, block_size=self.block_size, cover_block_rows=cover_bh, cover_block_cols=cover_bw, plan_blob=plan_blob)
        header_chunks = bytes_to_2bit_chunks(header_bytes)
        header_idx = pixel_order(total_pixels, len(header_chunks), key=b'public', tag=HEADER_TAG)
        cost_map = plan['distortion_cost_map']
        bpp_map = plan['bpp_map']
        self._log(f'  Cost range: [{cost_map.min():.4f}, {cost_map.max():.4f}]')
        self._log(f'  Cost mean: {cost_map.mean():.4f}')
        self._log(f'  Cost std: {cost_map.std():.4f}')
        self._log(f'  Blocks @ {PAYLOAD_BPP_HIGH} bpp (textured): {int(np.sum(bpp_map == PAYLOAD_BPP_HIGH))} / {bpp_map.size}')
        self._log(f'  Blocks @ {PAYLOAD_BPP_LOW} bpp (smooth):   {int(np.sum(bpp_map == PAYLOAD_BPP_LOW))} / {bpp_map.size}')
        self._step_header('Block-wise Embedding Capacity')
        capacities = plan['block_capacities']
        active = int(np.sum(capacities > 0))
        total_blocks = int(capacities.size)
        skipped = total_blocks - active
        self._log(f'  Total blocks: {total_blocks}')
        self._log(f'  Active blocks: {active} ({100 * active / total_blocks:.1f}%)')
        self._log(f'  Skipped blocks: {skipped} ({100 * skipped / total_blocks:.1f}%)')
        self._log(f'  Payload bits to embed: {total_payload_bits:,}')
        self._log(f'  Allocated capacity: {int(plan["bit_allocation"].sum()):,} bits ({int(capacities.sum()):,} pixels)')
        if active > 0:
            self._log(f'  Avg pixels/active block: {capacities[capacities > 0].mean():.1f}')
            self._log(f'  Max pixels/block: {int(capacities.max())}')
        self._step_header('Keyed Random Pixel Permutation (PRNG)')
        payload_idx = plan_to_pixel_indices(plan, (cover_h, cover_w), key=env['salt'], tag=PAYLOAD_TAG, header_indices=header_idx, block_size=self.block_size)                                                       
        pixel_bpp_full = np.repeat(np.repeat(bpp_map, self.block_size, axis=0), self.block_size, axis=1)[:cover_h, :cover_w].flatten()
        payload_bpp = pixel_bpp_full[payload_idx]
        cum_bits = np.cumsum(payload_bpp.astype(np.int64))
        if len(cum_bits) == 0 or cum_bits[-1] < total_payload_bits:
            available = int(cum_bits[-1]) if len(cum_bits) else 0
            raise ValueError(f'Adaptive plan produced only {available} payload bits but need {total_payload_bits}. Cover image too small.')
        cutoff = int(np.searchsorted(cum_bits, total_payload_bits, side='left')) + 1
        payload_idx = payload_idx[:cutoff]
        payload_bpp = payload_bpp[:cutoff]
        self._log(f'  Header pixel slots: {len(header_idx):,}')
        self._log(f'  Payload pixel slots: {len(payload_idx):,} (avg {payload_bpp.mean():.3f} bits/pixel)')
        self._log(f'  Total modified pixels: {len(header_idx) + len(payload_idx):,} / {total_pixels:,} ({100 * (len(header_idx) + len(payload_idx)) / total_pixels:.2f}%)')
        if len(payload_idx) > 0:
            self._log(f'  Payload pixel span: [{payload_idx.min():,}, {payload_idx.max():,}]')
        cap = capacity_bytes(cover_w, cover_h, len(header_bytes))
        self._step_header('Adaptive LSB Matching Embedding')
        payload_bits = bytes_to_bitarray(env['ciphertext'])
        payload_targets = pack_bits_to_symbols(payload_bits, payload_bpp)
        payload_modulus = (1 << payload_bpp.astype(np.int16)).astype(np.int16)
        stego_pixels = cover_pixels.copy()
        stego_pixels[header_idx] = lsb_match_encode(stego_pixels[header_idx], header_chunks)
        stego_pixels[payload_idx] = lsb_match_encode(stego_pixels[payload_idx], payload_targets, modulus=payload_modulus)
        self._log(f'  Header embedded: {len(header_chunks):,} chunks @ {BITS_PER_PIXEL} bpp (fixed)')
        self._log(f'  Payload embedded: {len(payload_targets):,} pixels, {total_payload_bits:,} bits, avg {payload_bpp.mean():.3f} bpp')
        self._step_header('Save Stego Image & Report Metrics')
        stego_arr = stego_pixels.reshape(cover_h, cover_w)
        stego_img = Image.fromarray(stego_arr)
        LOSSY_EXTS = {'.jpg', '.jpeg', '.webp'}
        out_ext = os.path.splitext(self.output_path)[1].lower()
        if out_ext in LOSSY_EXTS:
            old_path = self.output_path
            self.output_path = os.path.splitext(self.output_path)[0] + '.png'
            self._log(f"  WARNING: '{old_path}' uses lossy {out_ext} format which would destroy embedded data.")
            self._log(f"           Saving as '{self.output_path}' (lossless PNG) instead.")
        stego_img.save(self.output_path)
        self._log(f'  Saved: {self.output_path}')
        pm = psnr_mse(cover_arr, stego_arr)
        s = ssim(cover_arr, stego_arr)
        ht = histogram_tests(cover_arr, stego_arr)
        ber = image_ber(cover_arr, stego_arr)
        self._log(f'\n  Imperceptibility metrics:')
        self._log(f"    PSNR: {pm['psnr_db']:.2f} dB")
        self._log(f"    MSE:  {pm['mse']:.4f}")
        self._log(f'    SSIM: {s:.6f}')
        self._log(f"    BER:  {ber['ber']:.6f} ({ber['bit_errors']:,}/{ber['total_bits']:,} bits)")
        self._log(f"    Chi-square: stat={ht['chi_square_stat']:.4f}, p={ht['chi_square_p']:.6f}")
        self._log(f"    KS test: stat={ht['ks_stat']:.6f}, p={ht['ks_p']:.6f}")
        elapsed = time.time() - t0
        self._log(f"\n{'=' * 60}")
        print(f'Embedding {os.path.basename(self.cover_path)} + {os.path.basename(self.secret_path)} took {elapsed:.2f}s')
        self._log(f"{'=' * 60}")
        return {'stego_image': stego_img, 'output_path': self.output_path, 'ciphertext_len': len(env['ciphertext']), 'header_len': len(header_bytes), 'plan_blob_len': len(plan_blob), 'capacity_bytes': cap, 'capacity_used_pct': 100.0 * len(env['ciphertext']) / cap, 'bits_per_pixel': float(payload_bpp.mean()) if len(payload_bpp) else 0.0, 'header_bits_per_pixel': BITS_PER_PIXEL, 'pixels_modified': len(header_idx) + len(payload_idx), 'total_pixels': total_pixels, 'block_size': self.block_size, 'blocks_used': active, 'blocks_total': total_blocks, 'psnr_db': pm['psnr_db'], 'mse': pm['mse'], 'ssim': s, 'ber': ber['ber'], 'feature_hash': feature_hash.hex(), 'elapsed_s': elapsed}

def embed_secret(cover_path: str, secret_path: str, master_secret: bytes, output_path: str, block_size: int=DEFAULT_BLOCK_SIZE, verbose: bool=True, bpp_tiers: list=None) -> dict:
    pipeline = EmbeddingPipeline(cover_path, secret_path, master_secret, output_path, block_size, verbose, bpp_tiers=bpp_tiers)
    return pipeline.run()

class ExtractionPipeline:

    def __init__(self, stego_path: str, master_secret: bytes, verbose: bool=True):
        self.stego_path = stego_path
        self.master_secret = master_secret
        self.verbose = verbose
        self._step = 0
        self.stego_img = None
        self.stego_arr = None
        self.stego_pixels = None
        self.total_pixels = 0
        self.header = None
        self.header_idx = None
        self.payload_idx = None
        self.ciphertext = None
        self.compressed_payload = None
        self.recovered_img = None
        self.recovered_arr = None
        self.musical_result = None
        self.version = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _step_header(self, title: str):
        self._step += 1
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f'  Step {self._step}: {title}')
            print(f"{'=' * 60}")

    def run(self) -> dict:
        t0 = time.time()
        self._load_stego_image()
        self._read_header()
        if self.version == 3:
            result = self._extract_v2()
        else:
            raise ValueError(f'Unsupported header version: {self.version}')
        elapsed = time.time() - t0
        self._log(f"\n{'=' * 60}")
        self._log(f'  Extraction completed in {elapsed:.2f}s')
        self._log(f"{'=' * 60}")
        return result

    def _load_stego_image(self):
        self._step_header('Load Stego Image')
        if not os.path.exists(self.stego_path):
            raise FileNotFoundError(f'Stego image not found: {self.stego_path}')
        self.stego_img = Image.open(self.stego_path).convert('L')
        stego_w, stego_h = self.stego_img.size
        self.stego_arr = np.array(self.stego_img, dtype=np.uint8)
        self.stego_pixels = self.stego_arr.flatten()
        self.total_pixels = self.stego_pixels.size
        self._log(f'  Image: {self.stego_path}')
        self._log(f'  Dimensions: {stego_w} x {stego_h}')
        self._log(f'  Total pixels: {self.total_pixels:,}')
        self._log(f'  Mode: Grayscale (L)')

    def _read_header(self):
        self._step_header('Read Embedded Header')
        v2_fixed_chunks = V2_FIXED_LEN * (8 // BITS_PER_PIXEL)
        header_idx_probe = pixel_order(self.total_pixels, v2_fixed_chunks, key=b'public', tag=HEADER_TAG)
        v2_fixed_raw = lsb_match_decode(self.stego_pixels[header_idx_probe])
        v2_fixed_bytes = chunks_to_bytes(v2_fixed_raw, V2_FIXED_LEN)
        self.version = v2_fixed_bytes[0]
        self._log(f'  Detected header version: {self.version}')
        if self.version != 3:
            raise ValueError(f'Unsupported header version: {self.version}')
        plan_blob_len = struct.unpack('>I', v2_fixed_bytes[-4:])[0]
        full_header_len = V2_FIXED_LEN + plan_blob_len
        self._log(f'  V2 fixed header: {V2_FIXED_LEN} bytes')
        self._log(f'  Plan blob: {plan_blob_len} bytes (compressed)')
        self._log(f'  Total header: {full_header_len} bytes')
        full_header_chunks = full_header_len * (8 // BITS_PER_PIXEL)
        self.header_idx = pixel_order(self.total_pixels, full_header_chunks, key=b'public', tag=HEADER_TAG)
        header_raw = lsb_match_decode(self.stego_pixels[self.header_idx])
        header_bytes = chunks_to_bytes(header_raw, full_header_len)
        self.header = unpack_header(header_bytes)
        self._log(f'\n  Recovered header fields:')
        self._log(f"    Salt: {self.header['salt'].hex()}")
        self._log(f"    Nonce: {self.header['nonce'].hex()}")
        self._log(f"    Payload length: {self.header['payload_len']:,} bytes")
        self._log(f"    Secret image: {self.header['width']} x {self.header['height']}")
        if self.version == 3:
            self._log(f"    Feature hash: {self.header['feature_hash'].hex()}")
            self._log(f"    Block size: {self.header['block_size']}")
            self._log(f"    Cover block grid: {self.header['cover_block_rows']} x {self.header['cover_block_cols']}")

    def _extract_v2(self) -> dict:
        h = self.header
        stego_h, stego_w = self.stego_arr.shape
        self._step_header('Deserialize Adaptive Embedding Plan')
        restored_plan = deserialize_plan(h['plan_blob'], h['cover_block_rows'], h['cover_block_cols'])
        capacities = restored_plan['block_capacities']
        active_blocks = int(np.sum(capacities > 0))
        total_blocks = int(capacities.size)
        self._log(f"  Block grid: {h['cover_block_rows']} x {h['cover_block_cols']} = {total_blocks} blocks")
        self._log(f'  Active blocks: {active_blocks} ({100 * active_blocks / total_blocks:.1f}%)')
        self._log(f'  Skipped blocks: {total_blocks - active_blocks} ({100 * (total_blocks - active_blocks) / total_blocks:.1f}%)')
        self._log(f'  Total capacity in plan: {int(capacities.sum()):,} pixels')
        self._step_header('Reconstruct Keyed Random Pixel Order')
        total_payload_bits = h['payload_len'] * 8
        self._log(f'  Payload bits needed: {total_payload_bits:,}')
        self.payload_idx = plan_to_pixel_indices(restored_plan, (stego_h, stego_w), key=h['salt'], tag=PAYLOAD_TAG, header_indices=self.header_idx, block_size=h['block_size'])
        bpp_map = restored_plan['bpp_map']
        pixel_bpp_full = np.repeat(np.repeat(bpp_map, h['block_size'], axis=0), h['block_size'], axis=1)[:stego_h, :stego_w].flatten()
        payload_bpp = pixel_bpp_full[self.payload_idx]
        cum_bits = np.cumsum(payload_bpp.astype(np.int64))
        cutoff = int(np.searchsorted(cum_bits, total_payload_bits, side='left')) + 1
        self.payload_idx = self.payload_idx[:cutoff]
        payload_bpp = payload_bpp[:cutoff]
        self._log(f'  Pixel indices generated: {len(self.payload_idx):,} (avg {payload_bpp.mean():.3f} bits/pixel)')
        self._log(f'  Pixels span: [{self.payload_idx.min():,}, {self.payload_idx.max():,}]')
        self._step_header('Adaptive LSB Matching Extraction')
        payload_modulus = (1 << payload_bpp.astype(np.int16)).astype(np.int16)
        payload_symbols = lsb_match_decode(self.stego_pixels[self.payload_idx], modulus=payload_modulus)
        payload_bits = unpack_symbols_to_bits(payload_symbols, payload_bpp, total_bits_needed=total_payload_bits)
        self.ciphertext = bitarray_to_bytes(payload_bits, h['payload_len'])
        self._log(f'  Extracted ciphertext: {len(self.ciphertext):,} bytes')
        self._log(f'  (includes 16-byte AES-GCM auth tag)')
        self._step_header('AES-256-GCM Decryption')
        self._log(f'  Deriving key from master secret + salt via HKDF-SHA256...')
        try:
            self.compressed_payload = decrypt_payload(self.ciphertext, h['salt'], h['nonce'], self.master_secret)
            self._log(f'  Decryption successful!')
            self._log(f'  Authentication tag: VERIFIED')
            self._log(f'  Decrypted payload: {len(self.compressed_payload):,} bytes (compressed)')
        except Exception as e:
            self._log(f'  DECRYPTION FAILED: {type(e).__name__}')
            self._log(f'  This means the stego image was tampered with,')
            self._log(f'  corrupted, or the wrong password was provided.')
            raise
        self._step_header('Lossless Decompression (zlib)')
        raw_bytes = zlib.decompress(self.compressed_payload)
        self._log(f'  Compressed: {len(self.compressed_payload):,} bytes')
        self._log(f'  Decompressed: {len(raw_bytes):,} bytes')
        ratio = len(raw_bytes) / len(self.compressed_payload) if self.compressed_payload else 0
        self._log(f'  Compression ratio: {ratio:.2f}x')
        self._step_header('Reconstruct Secret Image')
        self.recovered_img = payload_to_image(self.compressed_payload, h['width'], h['height'])
        self.recovered_arr = np.array(self.recovered_img, dtype=np.uint8)
        self._log(f"  Reconstructed: {h['width']} x {h['height']} grayscale")
        self._log(f'  Pixel range: [{self.recovered_arr.min()}, {self.recovered_arr.max()}]')
        self._log(f'  Mean intensity: {self.recovered_arr.mean():.1f}')
        self._step_header('Generate Musical Feature Map')
        self.musical_result = compute_musical_energy_map(self.recovered_arr, h['block_size'])
        freq_map = self.musical_result['frequency_map']
        energy_map = self.musical_result['energy_map']
        bh, bw = self.musical_result['block_grid_shape']
        self._log(f'  Block grid: {bw} x {bh} = {bw * bh} blocks')
        self._log(f'  Frequency range: [{freq_map.min():.1f}, {freq_map.max():.1f}] Hz')
        self._log(f'  Energy range: [{energy_map.min():.3f}, {energy_map.max():.3f}]')
        self._log(f'  Energy mean: {energy_map.mean():.3f}')
        self._log(f'\n  Sample block frequencies:')
        sample_count = min(6, bh * bw)
        for idx in range(sample_count):
            bi, bj = np.unravel_index(idx, (bh, bw))
            self._log(f'    Block({bi},{bj}) -> {freq_map[bi, bj]:.1f} Hz (energy: {energy_map[bi, bj]:.3f})')
        self._step_header('Verify Musical Feature Consistency')
        recovered_hash = compute_feature_hash(energy_map)
        original_hash = h['feature_hash']
        hash_match = recovered_hash == original_hash
        self._log(f'  Original hash:  {original_hash.hex()}')
        self._log(f'  Recovered hash: {recovered_hash.hex()}')
        if hash_match:
            self._log(f'  Status: MATCH -- Musical feature integrity verified!')
        else:
            self._log(f'  Status: MISMATCH -- Possible data corruption!')
        self._log(f'\n  Cryptographic verification: PASSED (AES-GCM auth tag)')
        self._log(f"  Musical feature verification: {('PASSED' if hash_match else 'FAILED')}")
        recheck_payload = zlib.compress(raw_bytes, level=9)
        payload_ber = bit_error_rate(self.compressed_payload, recheck_payload)
        self._log(f"\n  Payload BER: {payload_ber['ber']:.6f} ({payload_ber['bit_errors']:,}/{payload_ber['total_bits']:,} bits)")
        return {'secret_bytes': self.compressed_payload, 'width': h['width'], 'height': h['height'], 'version': 3, 'recovered_image': self.recovered_img, 'feature_hash_match': hash_match, 'feature_hash_original': original_hash.hex(), 'feature_hash_recovered': recovered_hash.hex(), 'musical_energy_map': energy_map, 'frequency_map': freq_map, 'ber': payload_ber['ber']}

def extract_secret(stego_path: str, master_secret: bytes, output_path: str=None, verbose: bool=True) -> dict:
    pipeline = ExtractionPipeline(stego_path, master_secret, verbose)
    result = pipeline.run()
    if output_path and result.get('recovered_image'):
        result['recovered_image'].save(output_path)
        if verbose:
            print(f'\n  Recovered secret saved to: {output_path}')
    return result

def cmd_embed(args):
    master_secret = _resolve_password(args)
    verbose = not args.quiet
    try:
        result = embed_secret(args.cover_image, args.secret_image, master_secret, args.output, block_size=args.block_size, verbose=verbose, bpp_tiers=args.bpp_tiers)
    except Exception as e:
        print(f'\nEmbedding FAILED: {type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(1)
    if verbose:
        print(f"\n{'=' * 60}")
        print(f'  EMBEDDING SUMMARY')
        print(f"{'=' * 60}")
        print(f"  Stego image: {result['output_path']}")
        print(f"  PSNR: {result['psnr_db']:.2f} dB | SSIM: {result['ssim']:.6f}")
        print(f"  Capacity used: {result['capacity_used_pct']:.2f}%")
        print(f"  Blocks: {result['blocks_used']}/{result['blocks_total']} active")
        print(f"  Feature hash: {result['feature_hash']}")

def cmd_extract(args):
    master_secret = _resolve_password(args)
    verbose = not args.quiet
    try:
        result = extract_secret(args.stego_image, master_secret, output_path=args.output, verbose=verbose)
    except Exception as e:
        print(f'\nExtraction FAILED: {type(e).__name__}: {e}', file=sys.stderr)
        sys.exit(1)
    if args.verify_against and os.path.exists(args.verify_against):
        print(f"\n{'=' * 60}")
        print(f'  Verification Against Original')
        print(f"{'=' * 60}")
        original_arr = np.array(Image.open(args.verify_against).convert('L'))
        recovered_arr = np.array(result['recovered_image'])
        if original_arr.shape == recovered_arr.shape:
            pm = psnr_mse(original_arr, recovered_arr)
            s = ssim(original_arr, recovered_arr)
            ber = image_ber(original_arr, recovered_arr)
            nc = ncc(original_arr, recovered_arr)
            print(f"  PSNR: {pm['psnr_db']:.2f} dB | SSIM: {s:.6f}")
            print(f"  BER:  {ber['ber']:.6f} ({ber['bit_errors']:,}/{ber['total_bits']:,} bits)")
            print(f'  NCC:  {nc:.6f}')
            print(f'  Pixel-perfect: {np.array_equal(original_arr, recovered_arr)}')
        else:
            print(f'  Shape mismatch: {original_arr.shape} vs {recovered_arr.shape}')
    if verbose:
        print(f"\n{'=' * 60}")
        print(f'  EXTRACTION SUMMARY')
        print(f"{'=' * 60}")
        print(f"  Version: v{result['version']}")
        print(f"  Secret: {result['width']} x {result['height']}")
        print(f'  Output: {args.output}')
        if result.get('feature_hash_match') is not None:
            s = 'PASSED' if result['feature_hash_match'] else 'FAILED'
            print(f'  Musical verification: {s}')
        print(f'  Crypto verification: PASSED')

def cmd_analyze(args):
    img = Image.open(args.image).convert('L')
    arr = np.array(img, dtype=np.uint8)
    w, h = img.size
    bs = args.block_size
    print(f"{'=' * 60}")
    print(f'  Image Analysis: {args.image}')
    print(f"{'=' * 60}")
    print(f'  Dimensions: {w} x {h}')
    print(f'  Block size: {bs} x {bs}')
    ca = analyze_cover(arr, bs)
    bh, bw = ca['block_grid_shape']
    print(f'\n  Cover Analysis ({bw} x {bh} = {bw * bh} blocks):')
    print(f"    Texture:  mean={ca['texture_map'].mean():.4f}  std={ca['texture_map'].std():.4f}  range=[{ca['texture_map'].min():.4f}, {ca['texture_map'].max():.4f}]")
    print(f"    Entropy:  mean={ca['entropy_map'].mean():.4f}  std={ca['entropy_map'].std():.4f}  range=[{ca['entropy_map'].min():.4f}, {ca['entropy_map'].max():.4f}]")
    print(f"    Edge:     mean={ca['edge_map'].mean():.4f}  std={ca['edge_map'].std():.4f}  range=[{ca['edge_map'].min():.4f}, {ca['edge_map'].max():.4f}]")
    musical = compute_musical_energy_map(arr, bs)
    freq = musical['frequency_map']
    energy = musical['energy_map']
    print(f'\n  Musical Features:')
    print(f'    Frequency range: [{freq.min():.1f}, {freq.max():.1f}] Hz')
    print(f'    Energy mean: {energy.mean():.4f}  std: {energy.std():.4f}')
    cap = capacity_bytes(w, h)
    print(f'\n  Embedding Capacity:')
    print(f'    Max payload: {cap:,} bytes ({cap / 1024:.1f} KB)')
    print(f'    Bits/pixel: {8 * cap / (w * h):.4f}')

def cmd_capacity(args):
    img = Image.open(args.cover_image).convert('L')
    w, h = img.size
    total = w * h
    bs = args.block_size
    bh = max(1, (h + bs - 1) // bs)
    bw = max(1, (w + bs - 1) // bs)
    est_plan_blob = int(6 * bh * bw * 0.4)
    est_header_v2 = V2_FIXED_LEN + est_plan_blob
    cap_v2 = capacity_bytes(w, h, est_header_v2)
    print(f"{'=' * 60}")
    print(f'  Capacity Report: {args.cover_image}')
    print(f"{'=' * 60}")
    print(f'  Image: {w} x {h} ({total:,} pixels)')
    print(f'  Block size: {bs} x {bs} ({bh} x {bw} = {bh * bw} blocks)')
    print(f'\n  Adaptive Capacity: ~{cap_v2:,} bytes (~{cap_v2 / 1024:.1f} KB)')
    print(f'  Header overhead: ~{est_header_v2:,} bytes')
    print(f'\n  Bits per pixel: ~{8 * cap_v2 / total:.4f}')
    est_secret_pixels = int(cap_v2 * 0.5 / 1)
    print(f'\n  Estimated max secret image:')
    side = int(est_secret_pixels ** 0.5)
    print(f'    ~{side} x {side} grayscale (depends on content compressibility)')

def _resolve_password(args) -> bytes:
    if args.password:
        return args.password.encode('utf-8')
    elif args.password_file:
        with open(args.password_file, 'r') as f:
            return f.readline().strip().encode('utf-8')
    else:
        print('Error: Provide --password or --password-file', file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(prog='steg', description='Adaptive Musical-Feature-Driven Steganography Tool', formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    p_embed = subparsers.add_parser('embed', help='Hide a secret image inside a cover image', formatter_class=argparse.RawDescriptionHelpFormatter)
    p_embed.add_argument('cover_image', help='Cover image (carrier)')
    p_embed.add_argument('secret_image', help='Secret image to hide')
    p_embed.add_argument('output', help='Output stego image path (PNG)')
    p_embed.add_argument('--password', '-p', help='Passphrase')
    p_embed.add_argument('--password-file', '-f', help='File with passphrase')
    p_embed.add_argument('--block-size', '-b', type=int, default=8, help='Block size (default: 8)')
    p_embed.add_argument('--bpp-tiers', type=parse_bpp_tiers, default=None, help='Percentile tiers for adaptive bpp, as "frac:bpp,frac:bpp,...", cumulative fractions ascending, ranked busiest-block-first. Default "0.80:2,0.95:1,1.0:0" -> busiest 80%% of blocks at 2 bpp, next 15%% at 1 bpp, smoothest 5%% skipped entirely.')
    p_embed.add_argument('--quiet', '-q', action='store_true', help='Suppress detailed output')
    p_extract = subparsers.add_parser('extract', help='Recover a hidden image from a stego image', formatter_class=argparse.RawDescriptionHelpFormatter)
    p_extract.add_argument('stego_image', help='Stego image to extract from')
    p_extract.add_argument('output', help='Output recovered image path')
    p_extract.add_argument('--password', '-p', help='Passphrase')
    p_extract.add_argument('--password-file', '-f', help='File with passphrase')
    p_extract.add_argument('--verify-against', help='Original secret for comparison')
    p_extract.add_argument('--quiet', '-q', action='store_true', help='Suppress detailed output')
    p_analyze = subparsers.add_parser('analyze', help='Analyze image characteristics')
    p_analyze.add_argument('image', help='Image to analyze')
    p_analyze.add_argument('--block-size', '-b', type=int, default=8, help='Block size (default: 8)')
    p_capacity = subparsers.add_parser('capacity', help='Report embedding capacity of a cover image')
    p_capacity.add_argument('cover_image', help='Cover image to check')
    p_capacity.add_argument('--block-size', '-b', type=int, default=8, help='Block size (default: 8)')
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(0)
    commands = {'embed': cmd_embed, 'extract': cmd_extract, 'analyze': cmd_analyze, 'capacity': cmd_capacity}
    commands[args.command](args)
if __name__ == '__main__':
    main()
