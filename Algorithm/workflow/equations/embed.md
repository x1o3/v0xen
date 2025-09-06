
## v0xen embedding algorithm

Let:

- $I \in \mathbb{Z}^{H \times W}$ : grayscale cover image of height $H$, width $W$
- $S \in \mathbb{Z}^{h \times w}$ : secret image (resized)
- $N = H \times W$ : total pixels

---

### 1. Preprocessing & Resizing

Resize $S$ to match half the width of $I$ and fit height accordingly:

$$ w' = \frac{W}{2}, \quad h' = \left\lfloor w' \cdot \frac{h}{w} \right\rfloor $$ $$ S' \in \mathbb{Z}^{h' \times w'} = \text{Resize}(S, w', h') $$

---

### 2. Extract Top-4 MSBs

For each pixel $p \in S'$:

$$ \text{MSB}_4(p) = (p \gg 4) \;\&\; 0b1111 $$

This gives an array:

$$ M = [m_{ij}] \in \{0,1,\dots,15\}^{h' \times w'} $$

---

### 3. Encoding to Musical Notes ---> Encoded array

*i.* Extend each 4-bit value $m$:

$$ m' = m \;||\; 1001 $$


*ii.* Convert to int (frequency) and map to musical notes :

$$ f = \text{int}(m', 2), \quad \text{Note} = \texttt{librosa.hz\_to\_note}(f) $$


*iii.* Convert `Note` → Hex → Binary via mapping $\mathcal{F}$:

$$
\text{Hex} = \text{encode\_to\_hex}(\text{Note})
$$

*iv.* Replacing the `#` in all Hex :
$$
\text{Hex}' = \text{Hex} \; \text{with } \# \to 69
$$

*v.* Convert Hex' → Binary:
$$
\text{Binary} = D[\text{Hex}']
$$

*vi.* Append '0001' if (Hex' length > 3):
$$
\text{Final Bits} =
\begin{cases}
\text{Binary} \;||\; 0001, & \text{if } |\text{Hex}'| > 3 \\
\text{Binary}, & \text{otherwise}
\end{cases}
$$

*vii.* Resulting encoded array :
$$
E = [e_{ij}] \in \{0,1,\dots,15\}^{h' \times 3w'}
$$

---

### 4. Mapping with Hash-Based Function

*i.* Define input tuples $\mathcal{T} = {t_1, t_2, \dots, t_K}$ (triplets).  

*ii.* Mapping function using SHA-256:

$$ \phi(t) = \Big( \text{SHA256}(k \;||\; t) \;\bmod\; 16 \Big) $$

Applied row-wise:
$$ M' = [\phi(e_{i,j:j+3})] $$

---

### 5. Bit Expansion

*i.* For each symbol $v \in M'$:

$$ \text{High}(v) = (v \gg 2) \;\&\; 0b11 $$ $$ \text{Low}(v) = v \;\&\; 0b11 $$ $$ F = [\text{High}(v), \text{Low}(v)] $$

*ii* Final array:

$$ F \in \{0,1,2,3\}^{h' \times 2w''} $$

---

### 6. Bit Encoding (XOR step)

For every **odd index** $i$:

$$ F_{i} \gets F_{i} \oplus 0b11 $$

---

### 7. Embedding into Cover Image

*i.* For each pixel $c \in I$:

$$ c' = (c \;\&\; 0b11111100) \;|\; (f \;\&\; 0b11) $$

*ii.* Stego-image:

$$ I' = \text{Embed}(I, F) $$

---

### Evaluation Metrics


#### *i.* PSNR (Peak Signal-to-Noise Ratio)

 **Mean Squared Error (MSE):**

$$
\text{MSE} = \frac{1}{MN} \sum_{i=1}^{M} \sum_{j=1}^{N} \big( I_1(i,j) - I_2(i,j) \big)^2
$$
Where:

- $I_1$ = Original image (cover)
- $I_2$ = Stego image
- $M, N$ = Image dimensions

 **PSNR:**

$$
\text{PSNR} = 20 \cdot \log_{10} \left( \frac{MAX_I}{\sqrt{\text{MSE}}} \right)
$$
Where $MAX_I$ is the maximum pixel value (usually 255).


---

#### *ii.* SSIM (Structural Similarity Index Measure)

For two windows $x$ and $y$:

$$
\text{SSIM}(x,y) = \frac{(2\mu_x \mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
$$

Where:

- $\mu_x, \mu_y$ = averages of $x,y$ 
- $\sigma_x^2, \sigma_y^2$ = variances
- $\sigma_{xy}$ = covariance
- $C_1, C_2$ = stability constants

---
#### *iii.* BER (Bit Error Rate)

$$
BER=∑i=1n(bo,i⊕br,i)n
$$
$$
\text{BER} = \frac{\sum_{i=1}^{n} (b_{o,i} \oplus b_{r,i})}{n}
$$
$$
BER=n∑i=1n​(bo,i​⊕br,i​)​
$$
Where:

- $b_o, b_r$ = binary arrays of original and received images
- $\oplus$ = XOR
- $n$ = total number of bits

---

#### *iv.* LPIPS (Learned Perceptual Image Patch Similarity)
$$
\text{LPIPS}(I,I') = \sum_{l} w_l \cdot \left\lVert \Phi(I) - \Phi(I') \right\rVert_2^2
$$
Where:
- $\Phi$ = feature maps from pretrained networks (e.g., AlexNet) 
- $w_l$ = learned weights

---


