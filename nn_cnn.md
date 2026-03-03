## Overview

CNNs are designed for spatially-structured data where nearby values are related (images, sensor grids, spatial maps).

Good for:
- Image classification and object detection
- Spatial sensor data (lidar, depth maps, sensor grids)
- Any grid-structured input where spatial relationships matter
- Pattern recognition in 2D/3D data
- Feature extraction from visual or spatial information

## Core Concept

CNN = Slide filters across input to detect patterns

Instead of connecting every input to every output (fully connected), CNNs:
1. Use small filters (e.g., 3×3) with learnable weights
2. Slide these filters across the entire input
3. Detect patterns wherever they appear
4. Stack multiple layers for hierarchical features

## Definition

For a 2D input $X \in \mathbb{R}^{H \times W}$ and filter $K \in \mathbb{R}^{k_h \times k_w}$:

$$(X * K)[i, j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X[i+m, j+n] \cdot K[m, n]$$

Where:
- $*$ denotes convolution operation
- $(i, j)$ is the output position
- $(m, n)$ iterates over the filter
- The filter slides across all valid positions

Multi-Channel Convolution:

For input $X \in \mathbb{R}^{C_{in} \times H \times W}$ and filter $K \in \mathbb{R}^{C_{out} \times C_{in} \times k_h \times k_w}$:

$$Y[c_{out}, i, j] = \sum_{c_{in}=0}^{C_{in}-1} \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X[c_{in}, i+m, j+n] \cdot K[c_{out}, c_{in}, m, n] + b[c_{out}]$$

Where:
- $C_{in}$ = number of input channels
- $C_{out}$ = number of output channels (number of filters)
- $b[c_{out}]$ = bias for output channel $c_{out}$

## Output Size Calculation

$$H_{out} = \left\lfloor \frac{H_{in} + 2 \times \text{padding}_h - k_h}{\text{stride}_h} \right\rfloor + 1$$

$$W_{out} = \left\lfloor \frac{W_{in} + 2 \times \text{padding}_w - k_w}{\text{stride}_w} \right\rfloor + 1$$

Where:
- $H_{in}, W_{in}$ = input height and width
- $k_h, k_w$ = kernel (filter) height and width
- $\text{padding}$ = number of zeros added to borders
- $\text{stride}$ = step size when sliding filter
- $\lfloor \cdot \rfloor$ = floor function (round down)

Example:

Input: 20×20, kernel=3×3, stride=2, padding=1

$$H_{out} = \left\lfloor \frac{20 + 2 \times 1 - 3}{2} \right\rfloor + 1 = \left\lfloor \frac{19}{2} \right\rfloor + 1 = 9 + 1 = 10$$

Output: 10×10

## Kernel Size (Filter Size)

What it is: Size of the sliding filter (e.g., 3×3, 5×5, 1×1)

Effect:
- Small kernels (1×1, 3×3): 
  - Fewer parameters
  - Can stack many layers
  - Standard choice
- Large kernels (7×7, 11×11):
  - More parameters
  - Larger receptive field per layer
  - Computationally expensive

Common choices: 3×3 (most popular), 1×1 (channel mixing), 5×5 (wider context)

## Stride

What it is: Step size when sliding the filter

Effect:
- stride=1: Move 1 pixel at a time → larger output, preserves spatial detail
- stride=2: Move 2 pixels at a time → output half the size, downsampling
- stride=(1, 2): Different strides for height and width

Usage:
- stride=1: Maintain spatial resolution
- stride≥2: Reduce spatial dimensions (alternative to pooling)

## Padding

What it is: Add zeros around input borders

Effect:
- padding=0: No padding → output smaller than input
- padding=1: Add 1 row/column of zeros → can maintain size with kernel=3, stride=1
- padding='same': Auto-calculate padding to keep output size = input size

Usage:
- Padding=0: When you want size reduction
- Padding>0: Preserve spatial dimensions, prevent information loss at borders

## Number of Filters (Output Channels)

What it is: How many different filters to learn

Effect:
- More filters: More diverse patterns detected, more parameters, richer features
- Fewer filters: Fewer parameters, less expressive

Common patterns:
- Start with 16-64 filters in early layers
- Increase in deeper layers (64 → 128 → 256)
- Or decrease for compression

## Receptive Field

Receptive field = the region of the input that influences one output neuron

Calculation for Stacked Layers:

For a sequence of convolutional layers:

$$RF_1 = k_1$$

$$RF_2 = RF_1 + (k_2 - 1) \times s_1$$

$$RF_3 = RF_2 + (k_3 - 1) \times s_1 \times s_2$$

Where:
- $k_i$ = kernel size of layer $i$
- $s_i$ = stride of layer $i$
- $RF_i$ = receptive field after layer $i$

Example: Three 3×3 Layers (stride=1)

- Layer 1: RF = 3
- Layer 2: RF = 3 + (3-1)×1 = 5
- Layer 3: RF = 5 + (3-1)×1 = 7

Key insight: Stacking small kernels achieves large receptive field efficiently

Why Receptive Field Matters:

- Small RF (3×3): Only sees local patterns → "edge here" or "corner detected"
- Large RF (11×11): Sees broader context → "object shape" or "scene layout"
- Larger RF = more context = better decisions

## MaxPooling

MaxPool = take the maximum value from each region

$$Y[i, j] = \max_{m=0}^{k-1} \max_{n=0}^{k-1} X[i \times s + m, j \times s + n]$$

Where:
- $k$ = pooling kernel size
- $s$ = pooling stride

Example:

```
Input (4×4):
[1 3 2 4]
[5 6 7 8]
[2 1 9 3]
[4 2 5 6]

MaxPool2d(kernel_size=2, stride=2):

Region 1:    Region 2:
[1 3]        [2 4]
[5 6]        [7 8]
max=6        max=8

Region 3:    Region 4:
[2 1]        [9 3]
[4 2]        [5 6]
max=4        max=9

Output (2×2):
[6 8]
[4 9]
```

Why Use MaxPooling:

1. Downsampling: Reduces spatial dimensions by factor of 2-3×
2. Translation invariance: Small shifts don't change output (max is still max)
3. Computational efficiency: Smaller inputs for subsequent layers
4. Feature robustness: Keeps strongest activations

When to Use:

- Early layers: Aggressive downsampling when input is large
- Not in later layers: Spatial size already small, don't want to over-compress
- Alternative: Use stride≥2 in conv layers instead

## Parameter Efficiency

Comparison: CNN vs Fully Connected

Fully Connected Layer:
- Input: 20×20 = 400 values
- Output: 128 neurons
- Parameters: 400 × 128 = 51,200 parameters

Convolutional Layer:
- Input: 20×20
- Filters: 8 filters of size 3×3
- Parameters: 8 × 3 × 3 = 72 parameters (plus 8 biases)

CNN is ~700× more parameter-efficient!

Why Stacking Small Kernels is Efficient:

Option A: One 11×11 layer
- Receptive field: 11×11
- Parameters (64 channels): 64 × 64 × 11 × 11 = 495,616

Option B: Five 3×3 layers
- Receptive field: 11×11 (same!)
- Parameters: 5 × (64 × 64 × 3 × 3) ≈ 184,320
- 2.7× fewer parameters
- More non-linearity (5 ReLUs vs 1)
- Better performance (empirically proven)

## Weight Sharing

The same filter is applied at every spatial position.

Example:
- Filter learns to detect "vertical edge"
- This same filter slides across entire input
- Can detect vertical edges anywhere (top, bottom, left, right)

Benefits:

1. Parameter efficiency: One filter (9 weights for 3×3) vs separate weights for each position (thousands)
2. Translation invariance: Pattern detected regardless of position
3. Generalization: Learn once, apply everywhere

## Design Patterns

Kernel sizes: The standard choice is 3×3 kernels (VGG, ResNet) for good balance of receptive field and efficiency. Stack multiple 3×3 layers instead of one large kernel. Use 1×1 for channel mixing or dimensionality reduction. Use 5×5 or 7×7 only for first layer when input resolution is very high.

Downsampling: Modern architectures prefer stride ≥ 2 (learnable) over pooling (fixed). MaxPool is still common for aggressive early downsampling and translation invariance.

Padding: Use padding=0 when downsampling is desired. Use padding='same' to preserve spatial dimensions with stride=1.

Channel progression: Typical pattern increases channels as spatial size decreases (3 → 64 → 128 → 256 → 512). Alternative pattern decreases both for compact representation.

Network depth: Small inputs (20×20 to 64×64) use 3-5 layers. Large inputs (224×224) use 10-100+ layers. Trade-off: more layers give larger receptive field and abstraction but are harder to train.

