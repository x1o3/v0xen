
# v0xen extraction algorithm

Let:

- $P_s \in \mathbb{Z}^{H \times W}$ : stego image 
- $h,w$ : dimensions of $P_s$  
- $I_s$ : reconstructed secret image  

---

## 1. Extract Embedded MSBs

For each pixel:

$$ M(x,y) = P_s(x,y) \;\&\; 0b00000011 $$

Resulting array:

$$ M \in \{0,1,2,3\}^{h \times w} $$

---

## 2. XOR Decode

For every odd index $i$ in each row:
$$
M[r,i] = M[r,i] \oplus 0b11,\qquad \forall\ i\ \text{odd}
$$
---

## 3. Array Conversion (Combine High + Low Bits)

*i.* Combining high bits with low bits :
$$ v = (M[y,2x] \ll 2) \;|\; M[y,2x+1] $$

*ii.* Saving in an array :
$$ A[y,x] = v $$ where 
$$ A \in [0,15]^{h \times (w/2)} $$ ---
## 4. Grouping into Triplets :

Group every 3 values from array $A$:

$$
T = (A[y,j], A[y,j+1], A[y,j+2]), \quad j = 0,3,6,\dots
$$

$$
G \in \{0,\dots,15\}^{h \times (w/6) \times 3}
$$

---

## 5. Dynamic key-based mapping:

#### *i.* Table Generation:

A secret key $k$ generates a unique mapping:

$$
\text{mapping}_k : T \to \{0,\dots,15\}
$$

Assignment rule (collision resolution ensuring bijection):

$$
\text{mapping}_k(t) = \big(\text{SHA256}(k \;||\; \text{str}(t))\big) \bmod 16
$$

---
#### *ii.* Extraction (inverse mapping)

For each identifier $a \in A$:
$$
t = \text{mapping}_k^{-1}(a), \qquad t \in T
$$

The recovered tuples are concatenated:
$$
T' = \bigcup_{a \in A} \text{mapping}_k^{-1}(a)
$$

where 
$$
T' \in \{0,\dots,15\}^{h \times (w/6) \times 3}
$$

is the expanded triplet sequence.

---

## 6. Convert Dynamic Hex → ASCII Notes

*i.* Flatten triplets:

$$
\text{HexTriplets} = \text{flatten}(T')
$$

*ii.* Normalize sharp symbols:

$$
\text{Hex}' = \text{HexTriplets} \;\text{with } 69 \to 266f
$$

*iii.* Convert to ASCII (skip undeclared values, e.g. `0001`):

$$
\text{ASCII} = \text{decode}(\text{Hex}')
$$

---
## 7. Extract musical notes via regex:

$$
\text{Notes} = \text{regex\_extract}(\text{ASCII})
$$
---

## 8. Convert Notes → Frequency

$$ f = \text{librosa.note\_to\_hz}(\text{Notes}) $$

Apply correction table (MANDATORY):

$$ f' = \text{replace}(f, \text{table}) $$

---

## 9. Convert Frequency → Binary

Convert frequency to 8-bit binary:

$$
b_f = \text{format}(f', 8)
$$

Final bits:

$$
B_f \in \{0,1\}^{h \times w}, \quad B_f[y,x] = b_f
$$

---

## 10. Secret Image Reconstruction


Mapping in bounds only:
$$
I_s(x,y) = 
\begin{cases}
\text{int}(B_f[y,x], 2), & \text{if } B_f[y,x] \text{ exists} \\
0, & \text{otherwise}
\end{cases}
$$

Secret Image:
$$ I_s \in [0,255]^{h' \times w'} $$

---

## Evaluation Metrics

#### *i.* Bit Error Rate (BER)

$$ BER = \frac{\sum_{i=1}^n (b_{o,i} \oplus b_{r,i})}{n} $$

---

#### *ii.* Normalized Cross-Correlation (NCC)

$$ NCC = \frac{\sum (O - \mu_O)(R - \mu_R)}{\sqrt{\sum (O - \mu_O)^2 \;\sum (R - \mu_R)^2}} $$

---
