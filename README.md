# [V0XEN](https://doi.org/10.1007/s40010-026-01193-0)

Python implementation of the **Music-Inspired Adaptive Technique for Secure and Imperceptible Image Steganography**.

This technique combines music-inspired feature representation, adaptive distortion-cost guided embedding, keyed pixel permutation, variable-bit LSB matching, and authenticated encryption to conceal a secret image inside a cover image.

> **Publication:** J. Kakkar and M. Dalal, *Music-Inspired Adaptive Technique for Secure and Imperceptible Image Steganography*, Proceedings of the National Academy of Sciences, India Section A: Physical Sciences, 2026 (SCIE).
>
> **DOI:** `10.1007/s40010-026-01193-0`

---

## Performance

| Metric |  result |
|---|---:|
| Average embedding capacity | **1.85 BPP** |
| PSNR | **> 47 dB** |
| SSIM | **> 0.995** |
| BER | **0%** |

---

## Usage

```sh
usage: steg [-h] {embed,extract,analyze,capacity} ...

positional arguments:
  {embed,extract,analyze,capacity}
                        Choose a mode
    embed               Embed a secret image into a cover image
    extract             Extract a secret image from a stego image
    analyze             Analyze image characteristics
    capacity            Report embedding capacity of a cover image
```

### embed:

```sh
usage: steg embed [-h] [--password PASSWORD] [--password-file PASSWORD_FILE]
                  [--block-size BLOCK_SIZE] [--bpp-tiers BPP_TIERS] [--quiet]
                  cover_image secret_image output

positional arguments:
  cover_image           Cover image path
  secret_image          Secret image path
  output                Stego image output path

options:
  -h, --help            show this help message and exit
  -p, --password        Passphrase
  -f, --password-file   File containing passphrase
  -b, --block-size      Block size (default: 8)
  --bpp-tiers           Adaptive BPP tiers
  -q, --quiet           Suppress detailed output
```


### extract:

```sh
usage: steg extract [-h] [--password PASSWORD] [--password-file PASSWORD_FILE]
                    [--verify-against VERIFY_AGAINST] [--quiet]
                    stego_image output

positional arguments:
  stego_image           Stego image path
  output                Extracted secret image path

options:
  -h, --help            show this help message and exit
  -p, --password        Passphrase
  -f, --password-file   File containing passphrase
  --verify-against      Original secret image for comparison
  -q, --quiet           Suppress detailed output
```

### analyze:

```sh
usage: steg analyze [-h] [--block-size BLOCK_SIZE] image

positional arguments:
  image                 Image path

options:
  -h, --help            show this help message and exit
  -b, --block-size      Block size (default: 8)
capacity:
usage: steg capacity [-h] [--block-size BLOCK_SIZE] cover_image

positional arguments:
  cover_image           Cover image path

options:
  -h, --help            show this help message and exit
  -b, --block-size      Block size (default: 8)
```

### Examples

```sh
python steg.py embed ./cover.png ./secret.png ./stego.png -p "my-secret"
python steg.py extract ./stego.png ./recovered.png -p "my-secret"
python steg.py extract ./stego.png ./recovered.png -p "my-secret" --verify-against ./secret.png
python steg.py analyze ./cover.png
python steg.py capacity ./cover.png
```

This repo includes an dataset/ directory with a few example cover, and secret images.
These images are provided for demonstration purposes only from these datasets:
[Landscape images](https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization), [Ariel images](https://sipi.usc.edu/database/database.php?volume=aerials)
