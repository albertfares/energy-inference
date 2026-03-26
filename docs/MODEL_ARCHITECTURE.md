# Model Architectures: Benchmarked Models Overview

> **Context:** This project benchmarks DL vision models for inference energy on edge hardware (Jetson/CPU/GPU).
> Understanding *how* these models differ architecturally explains *why* they have different latency, FLOPs, and energy profiles.

---

## Background: What All These Models Share

All models here are **image classifiers or object detectors** built from layers of parameterized operations. At a high level, every forward pass:

1. Takes an input tensor of shape `[B, C, H, W]`
2. Transforms it through a series of operations (convolutions, attention, etc.)
3. Outputs class logits or bounding box predictions

The differences lie in **what those intermediate operations are**, **how deep the network is**, and **how information flows through it**.

---

## Part 1 — CNN-based Models

Convolutional Neural Networks (CNNs) apply learned filters over local spatial regions. A convolution with kernel size `k×k` over a `C_in`-channel feature map produces a `C_out`-channel output by computing:

```
output[c_out, h, w] = sum over (c_in, kh, kw) of: weight[c_out, c_in, kh, kw] * input[c_in, h+kh, w+kw]
```

FLOPs for one conv layer ≈ `2 × C_in × C_out × k² × H_out × W_out`

---

### VGG16

**Design philosophy:** depth through simplicity.

VGG stacks 13 convolutional layers (all 3×3, stride 1) followed by 3 fully-connected layers. Every block doubles the number of channels while halving spatial resolution via max-pooling.

```
Input → [Conv3×3 × 2] → Pool → [Conv3×3 × 2] → Pool → [Conv3×3 × 3] → Pool
      → [Conv3×3 × 3] → Pool → [Conv3×3 × 3] → Pool → FC(4096) → FC(4096) → FC(1000)
```

- **~138M parameters**, mostly in the FC layers
- **No skip connections, no bottlenecks** — every layer is fully computed
- Uniform and easy to reason about, but extremely compute- and memory-heavy
- The FC layers alone account for ~100M parameters, making it impractical for edge use

---

### GoogLeNet (Inception v1)

**Design philosophy:** why choose one filter size when you can use all of them?

GoogLeNet introduced the **Inception module**: instead of one conv operation per layer, it applies 1×1, 3×3, and 5×5 convolutions **in parallel**, concatenates the results, and lets the network learn which scale matters.

```
         ┌──── 1×1 conv ────────────────────┐
input ───┼──── 1×1 conv → 3×3 conv ─────────┼──→ concat → next layer
         ├──── 1×1 conv → 5×5 conv ─────────┤
         └──── MaxPool  → 1×1 conv ──────────┘
```

The 1×1 convolutions before 3×3 and 5×5 are **bottleneck projections** — they reduce channel depth before the expensive operations, cutting FLOPs significantly.

- **~6.8M parameters** — 20× fewer than VGG
- Global average pooling replaces large FC layers at the end
- More complex control flow than VGG, but much more parameter-efficient

---

### ResNet18 / ResNet50

**Design philosophy:** solve the degradation problem with identity shortcuts.

As networks get deeper, gradients vanish during backprop and accuracy paradoxically degrades. ResNet fixes this with **residual connections**: each block computes a residual `F(x)` and adds it to the input:

```
output = F(x) + x     ← the skip connection
```

This ensures gradients always have a direct path back, enabling networks with 50–150+ layers.

**ResNet18** uses **basic blocks** — two 3×3 convolutions per block:
```
x → Conv3×3 → BN → ReLU → Conv3×3 → BN → (+x) → ReLU
```

**ResNet50** uses **bottleneck blocks** — 1×1 to compress, 3×3 to convolve, 1×1 to expand:
```
x → Conv1×1(64) → BN → ReLU → Conv3×3(64) → BN → ReLU → Conv1×1(256) → BN → (+x) → ReLU
```

| | ResNet18 | ResNet50 |
|---|---|---|
| Params | ~11M | ~25M |
| Block type | Basic (3×3, 3×3) | Bottleneck (1×1, 3×3, 1×1) |
| Depth | 18 layers | 50 layers |
| FLOPs (224×224) | ~1.8G | ~4.1G |

---

## Part 2 — Efficient Mobile Architectures

Standard convolutions are expensive. These models replace them with cheaper factorized operations, making them viable for edge inference.

---

### Depthwise Separable Convolution (the shared building block)

A standard conv mixes spatial and channel information simultaneously.
A **depthwise separable conv** splits this into two steps:

1. **Depthwise conv:** one filter per input channel — captures spatial structure
2. **Pointwise conv (1×1):** mixes channels — captures cross-channel relationships

FLOPs reduction factor ≈ `1/C_out + 1/k²` — typically **8–9× cheaper** than standard conv for k=3.

---

### MobileNetV3 (Large / Small)

MobileNetV3 builds on depthwise separable convolutions and adds two more ideas:

**Squeeze-and-Excitation (SE):** a small gating mechanism that recalibrates channel importance:
```
x → GlobalAvgPool → FC → ReLU → FC → Sigmoid → scale × x
```
This lets the network "focus" on the most informative channels with minimal overhead.

**H-Swish activation:** a hardware-friendly approximation of Swish:
```
h_swish(x) = x × ReLU6(x + 3) / 6
```
Avoids the expensive exponential in standard Swish.

The `Large` variant has more layers and channels; the `Small` variant aggressively trims both. Both are designed to hit a specific FLOPs/accuracy operating point.

| | MobileNetV3-Large | MobileNetV3-Small |
|---|---|---|
| Params | ~5.4M | ~2.5M |
| FLOPs (224×224) | ~219M | ~56M |

---

### ShuffleNet V2

ShuffleNet makes a key observation: **FLOPs are a poor proxy for actual runtime** because memory access cost (MAC) matters too. It derives four practical efficiency guidelines and builds a block that obeys all of them.

The core trick: **channel split + shuffle**
At each block, channels are split into two halves. One half passes through unchanged (cheap identity branch). The other goes through a lightweight conv branch. They're concatenated, then channels are **shuffled** to mix information:

```
input → split → [identity branch | conv branch] → concat → channel shuffle → output
```

- Shuffle ensures information from both halves reaches all channels in the next layer
- No expensive group convolutions (avoids the MAC bottleneck of ShuffleNet v1)
- ~2.3M parameters, very fast on ARM hardware

---

## Part 3 — Vision Transformers

Transformers, originally from NLP, apply **self-attention** instead of convolutions. Rather than operating on local neighborhoods, attention computes relationships between **all positions simultaneously**.

Self-attention for a sequence of tokens:
```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
```
where Q, K, V are linear projections of the input. Every token attends to every other token — O(N²) complexity in sequence length N.

---

### ViT-B/16 (Vision Transformer)

ViT adapts the transformer encoder directly to images by treating **image patches as tokens**.

```
Image (224×224) → split into 14×14 = 196 patches of 16×16px
                → flatten each patch → linear embedding → sequence of 196 tokens
                → prepend [CLS] token → add positional encoding
                → 12 transformer encoder blocks
                → MLP head on [CLS] token → class logits
```

Each transformer block:
```
x → LayerNorm → Multi-Head Self-Attention → (+x) → LayerNorm → MLP → (+x)
```

**Key properties:**
- Global receptive field from layer 1 (every patch attends to every patch)
- No inductive bias for locality or translation equivariance — needs large data to compensate
- Fixed patch grid means input resolution must be a multiple of 16 (explains benchmark failures at odd resolutions)
- ~86M parameters, high FLOPs — not edge-friendly

---

### Swin Transformer

Swin addresses two ViT limitations: the O(N²) attention cost and the lack of a hierarchical feature map.

**Key idea: Window-based local attention + shifted windows**

Instead of global attention, each layer computes attention only within fixed local windows (e.g., 7×7 patches). This reduces complexity from O(N²) to O(N) in image size.

To allow cross-window communication, windows are **shifted by half their size** in alternating layers:

```
Layer L:   [window1][window2][window3]...     ← regular windows
Layer L+1: [  shifted windows overlap  ]...  ← shifted, enabling cross-window interaction
```

Swin also builds a **hierarchical feature pyramid** (like CNNs) by merging patches across stages, producing 4× downsampled feature maps at each stage. This makes it suitable for dense tasks (detection, segmentation).

| | ViT-B/16 | Swin-T |
|---|---|---|
| Params | ~86M | ~28M |
| Attention scope | Global | Local windows |
| Feature map | Single scale | Multi-scale |
| Resolution flexibility | Fixed multiples of 16 | Flexible |

---

## Part 4 — Object Detection Models

The models above are classifiers — they output a single label per image. Detection models also **localize** objects with bounding boxes.

---

### SSDLite

SSD (Single Shot Detector) adds detection heads directly onto a backbone's intermediate feature maps, predicting boxes and classes at **multiple scales simultaneously**.

SSDLite replaces all standard convolutions in the prediction heads with depthwise separable convolutions, using MobileNetV2/V3 as the backbone.

```
Image → MobileNet backbone → feature maps at 6 scales
                           → depthwise sep. conv heads
                           → [class scores + box offsets] per anchor
                           → NMS → final detections
```

- Efficient: inherits MobileNet's lightweight design throughout
- Trades some accuracy for speed and energy

---

### YOLO (You Only Look Once)

YOLO frames detection as a single regression problem. The image is divided into an S×S grid; each cell predicts B bounding boxes and class probabilities in **one forward pass**.

```
Image → Backbone (DarkNet / CSP / etc.) → Neck (FPN) → Detection head
      → [S × S × (B × 5 + C)] output tensor → NMS → final detections
```

Later versions (v5+) add:
- **CSP (Cross-Stage Partial) bottlenecks** for efficient feature reuse
- **Anchor-free heads** (v8+) predicting center/width/height directly
- **Feature Pyramid Network (FPN)** for multi-scale detection

YOLO is optimized for real-time inference and is widely used in production edge deployments.

---

## Summary Table

| Model | Params | FLOPs @224² | Core Mechanism | Multi-scale | Edge-Friendly |
|---|---|---|---|---|---|
| VGG16 | ~138M | ~15.5G | Uniform 3×3 conv stacks | No | No |
| GoogLeNet | ~6.8M | ~1.5G | Parallel Inception modules | No | Moderate |
| ResNet18 | ~11M | ~1.8G | Residual basic blocks | No | Yes |
| ResNet50 | ~25M | ~4.1G | Residual bottleneck blocks | No | Moderate |
| MobileNetV3-L | ~5.4M | ~219M | Depthwise sep. + SE + H-Swish | No | Yes |
| MobileNetV3-S | ~2.5M | ~56M | Depthwise sep. + SE + H-Swish | No | Yes |
| ShuffleNet V2 | ~2.3M | ~146M | Channel split + shuffle | No | Yes |
| ViT-B/16 | ~86M | ~17.6G | Global patch self-attention | No | No |
| Swin-T | ~28M | ~4.5G | Shifted window attention | Yes | Moderate |
| SSDLite | ~3.4M | ~300M | MobileNet + depthwise heads | Yes | Yes |
| YOLO | varies | ~medium | One-pass grid regression | Yes | Moderate |

---

## Relevance to Energy Inference

These models differ not just in FLOPs but in **arithmetic intensity** (FLOPs per byte of memory accessed):

- **Depthwise convolutions** (MobileNet, ShuffleNet) are memory-bound — low arithmetic intensity
- **Dense conv / FC layers** (VGG, ResNet) are compute-bound — high arithmetic intensity
- **Attention** (ViT, Swin) scales with sequence length and has irregular memory access patterns

This means FLOPs alone is not sufficient to predict energy. The energy predictor needs to capture these structural differences, which is why model family, parameter count, and MACs together are more informative than any single feature.
