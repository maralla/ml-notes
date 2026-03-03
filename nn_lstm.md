## Overview

LSTMs are designed for **sequential data with long-term dependencies** where current decisions depend on events many timesteps ago.

**Good for:**
- Time series forecasting
- Natural language processing (translation, sentiment analysis)
- Speech recognition
- Sequential decision making (RL, robot control)
- Any task requiring memory of 10-100+ timesteps

## Definition

### Inputs
- $x_t \in \mathbb{R}^{d_{in}}$: Current input
- $h_{t-1} \in \mathbb{R}^{d_h}$: Previous hidden state
- $c_{t-1} \in \mathbb{R}^{d_h}$: Previous cell state

### Parameters (Learned)
- $W_f, W_i, W_o, W_c \in \mathbb{R}^{d_h \times (d_h + d_{in})}$: Weight matrices
- $b_f, b_i, b_o, b_c \in \mathbb{R}^{d_h}$: Bias vectors

### Forward Pass Equations

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

$$h_t = o_t \odot \tanh(c_t)$$

### Notation
- $\sigma(z) = \frac{1}{1 + e^{-z}}$: Sigmoid function, range $(0, 1)$
- $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$: Hyperbolic tangent, range $(-1, 1)$
- $\odot$: Element-wise (Hadamard) multiplication
- $[h_{t-1}, x_t]$: Vector concatenation

---

## Gate Functions

### 1. Forget Gate ($f_t$)
**Role:** Controls retention of previous cell state

$$c_t = \underbrace{f_t \odot c_{t-1}}_{\text{keep old memory}} + i_t \odot \tilde{c}_t$$

- $f_t \approx 1$: Keep old memory (no forgetting)
- $f_t \approx 0$: Erase old memory (complete forgetting)
- $f_t \approx 0.5$: Keep half of old memory

**Note:** Despite the name, the forget gate can learn to NOT forget by outputting values close to 1.

### 2. Input Gate ($i_t$)
**Role:** Controls how much new information to add

$$c_t = f_t \odot c_{t-1} + \underbrace{i_t \odot \tilde{c}_t}_{\text{add new info}}$$

- $i_t \approx 1$: Fully incorporate new candidate
- $i_t \approx 0$: Ignore new information
- $i_t \approx 0.7$: Add 70% of new candidate

### 3. Cell Candidate ($\tilde{c}_t$)
**Role:** Computes new content to potentially add

- Uses $\tanh$ to produce values in $[-1, 1]$
- Creates the actual content (not just a gate)
- Modulated by input gate before being added

### 4. Output Gate ($o_t$)
**Role:** Controls what to expose from cell state

$$h_t = o_t \odot \tanh(c_t)$$

- $o_t \approx 1$: Fully expose cell state
- $o_t \approx 0$: Hide cell state (memory stored but not output)
- Allows information to be stored in $c_t$ without affecting $h_t$

---

## How Memory is Maintained

### The 48-Dimensional State Vector

For an LSTM with `hidden_size=48`:
- State is NOT storing 200 raw observations
- State is a **compressed representation** of temporal patterns
- Each dimension encodes abstract concepts learned during training

### Compression Strategy

**Instead of storing raw data:**
```
Timestep 180: input = [0.52, 0.31, 0.15, ...]
Timestep 181: input = [0.53, 0.30, 0.15, ...]
...
```

**LSTM encodes patterns:**
```
Dimension 7: "increasing trend detected" = 0.73
Dimension 8: "rate of change" = 0.21
Dimension 12: "periodic pattern active" = 0.85
```

### Temporal Decay

Information naturally decays over time:

- Recent timesteps: strong influence
  $$\text{weight} \approx 1.0$$

- Older timesteps: weak influence
  $$\text{weight} \approx 0.1$$

- Very old timesteps: negligible influence
  $$\text{weight} \approx 0.01$$

The LSTM doesn't need to remember all timesteps equally - older information contributes less to current decisions.

### Example: Tracking a Pattern

**Timestep 0-10:** Baseline state

$$c[12] = 0.05$$

**Timestep 11:** Pattern detected

$$f_t[12] = 0.90, \quad i_t[12] = 0.80, \quad \tilde{c}_t[12] = 0.95$$

$$c_t[12] = 0.90 \times 0.05 + 0.80 \times 0.95 = 0.805$$

**Timestep 12-20:** Pattern continues

$$f_t[12] \approx 0.95 \text{ (high retention)}$$

$$c_t[12] \text{ gradually increases: } 0.805 \to 0.88 \to 0.91 \to 0.93$$

**Timestep 21:** Pattern fading

$$f_t[12] = 0.70 \text{ (start forgetting)}$$

$$c_t[12] = 0.70 \times 0.93 + 0.20 \times 0.30 = 0.711$$

**Timestep 30:** Back to baseline

$$f_t[12] \to 0.10, \quad c_t[12] \to 0.05$$

---

## Training Process

### 1. Initialization
All weight matrices start with random values:
$$W_f, W_i, W_o, W_c \sim \mathcal{N}(0, 0.1)$$

Biases typically initialized to zero or small constants.

### 2. Forward Pass
Process input sequence through LSTM:
$$x_1, x_2, \ldots, x_T \to h_1, h_2, \ldots, h_T$$

Generate predictions from hidden states:

$$\hat{y}_t = f(h_t)$$

where $f$ is an output layer (linear layer for regression, softmax for classification).

### 3. Loss Computation

**For Classification (e.g., next word prediction):**

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{k=1}^{K} y_t^{(k)} \log \hat{y}_t^{(k)}$$

where $K$ is the number of classes.

**For Regression (e.g., time series forecasting):**

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} (y_t - \hat{y}_t)^2$$

### 4. Backpropagation Through Time (BPTT)
Compute gradients for all parameters by unrolling the network through time:

$$\frac{\partial \mathcal{L}}{\partial W_f}, \quad \frac{\partial \mathcal{L}}{\partial W_i}, \quad \frac{\partial \mathcal{L}}{\partial W_o}, \quad \frac{\partial \mathcal{L}}{\partial W_c}$$

Gradients flow backward through the sequence, with the cell state path preventing vanishing gradients.

### 5. Weight Update
Apply optimizer (SGD, Adam, etc.):

$$W \leftarrow W - \alpha \frac{\partial \mathcal{L}}{\partial W}$$

where $\alpha$ is the learning rate.

### 6. Iteration
Repeat over training dataset for multiple epochs until convergence.

---

## Why Gates Have Their Specific Meanings

The meaning of each weight matrix is **hardcoded by the architecture**, not learned.

### Architectural Constraints

**Cell State Update (Fixed Equation):**
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

This equation **forces** the roles:

1. **$f_t$ MUST control retention** because it multiplies $c_{t-1}$
   - Architecture: $f_t \odot c_{t-1}$
   - If $f_t = 1$: keep all old memory
   - If $f_t = 0$: erase old memory
   - $W_f$ cannot do anything else - it's constrained by this multiplication

2. **$i_t$ MUST control new information** because it multiplies $\tilde{c}_t$
   - Architecture: $i_t \odot \tilde{c}_t$
   - If $i_t = 1$: fully add new candidate
   - If $i_t = 0$: ignore new information
   - $W_i$ cannot do anything else

3. **$o_t$ MUST control exposure** because it multiplies $\tanh(c_t)$
   - Architecture: $h_t = o_t \odot \tanh(c_t)$
   - If $o_t = 1$: fully expose cell state
   - If $o_t = 0$: hide cell state
   - $W_o$ cannot do anything else

### What Learning Determines

Learning doesn't decide the role of each matrix. Learning decides:

- **WHEN** to forget (what patterns trigger high/low $f_t$)
- **WHAT** to add (what patterns trigger high/low $i_t$)
- **WHEN** to expose (what patterns trigger high/low $o_t$)
- **WHAT CONTENT** to create (what patterns produce specific $\tilde{c}_t$ values)

The roles are architectural; the policies are learned.

---

## How LSTMs Maintain Memory

### Not Storage, But Compression

An LSTM with `hidden_size=48` processing sequences of length 200 does NOT store 200 timesteps explicitly.

**Storage would require:**

$$200 \times d_{\text{input}} \text{ dimensions}$$

**LSTM uses:**

$$48 \text{ dimensions}$$

### Compression Mechanism

The 48-dimensional state encodes **patterns and abstractions**, not raw data:

**Example encodings (hypothetical):**
- Dimension 1-5: Current trend direction
- Dimension 6-10: Pattern phase tracking
- Dimension 11-15: Recent anomaly detection
- Dimension 16-20: Periodic cycle state
- ...

Each dimension is a continuous value encoding abstract temporal concepts.

### Information Flow Over Time

**Selective retention through gates:**

For dimension $j$ at timestep $t$:

$$c_t[j] = f_t[j] \cdot c_{t-1}[j] + i_t[j] \cdot \tilde{c}_t[j]$$

**Long-term memory:** Minimal decay

$$f_t[j] \approx 0.95\text{-}0.99$$

**Medium-term memory:** Gradual decay

$$f_t[j] \approx 0.7\text{-}0.9$$

**Short-term memory:** Rapid decay

$$f_t[j] \approx 0.1\text{-}0.5$$

### Effective Memory Span

With `hidden_size=48` and training sequences of 200 timesteps:
- **Theoretical maximum:** 200 timesteps
- **Practical effective span:** 20-50 timesteps
- **Recent timesteps:** Strong influence
- **Distant timesteps:** Weak influence (exponential decay)

---

## Weight Matrices Explained

### Dimensions

For `hidden_size=h` and `input_size=d`:

$$W_f, W_i, W_o, W_c \in \mathbb{R}^{h \times (h + d)}$$

Each matrix has learnable parameters:

$$h \times (h + d)$$

### What Each Matrix Does

All four matrices are **learned multiplier generators**:

1. **$W_f$**: Produces multipliers for old cell state
   - Learns: "In situation X, keep/forget dimension Y"
   
2. **$W_i$**: Produces multipliers for new candidate
   - Learns: "In situation X, add/ignore new information for dimension Y"
   
3. **$W_o$**: Produces multipliers for cell state exposure
   - Learns: "In situation X, expose/hide dimension Y"
   
4. **$W_c$**: Produces new content values
   - Learns: "In situation X, create content Y"

### Example: Single Dimension Update

For dimension $j$, the forget gate value is:

$$f_t[j] = \sigma\left(\sum_{k=1}^{h+d} W_f[j,k] \cdot z_t[k] + b_f[j]\right)$$

where the concatenated input is:

$$z_t = [h_{t-1}, x_t]$$

Each weight controls how much input feature $k$ influences the forget decision for dimension $j$:

$$W_f[j,k]$$

---

## Gradient Flow and Learning

### Why LSTM Solves Vanishing Gradients

**Regular RNN gradient:**

$$\frac{\partial h_T}{\partial h_0} = \prod_{t=1}^{T} \left(W_h \cdot \tanh'(\cdot)\right)$$

Product of many terms less than 1 causes vanishing:

$$\text{term} < 1$$

**LSTM gradient:**

$$\frac{\partial c_t}{\partial c_{t-1}} = f_t$$

$$\frac{\partial c_T}{\partial c_0} = \prod_{t=1}^{T} f_t$$

If forget gate is close to 1, gradient doesn't vanish:

$$f_t \approx 1$$

### How Weights Are Learned

**Step 1:** Forward pass produces predictions

**Step 2:** Compare predictions to ground truth labels

**Step 3:** Compute loss (how wrong were the predictions?)

**Step 4:** Backpropagation computes gradients:

$$\frac{\partial \mathcal{L}}{\partial W_f}, \quad \frac{\partial \mathcal{L}}{\partial W_i}, \quad \frac{\partial \mathcal{L}}{\partial W_o}, \quad \frac{\partial \mathcal{L}}{\partial W_c}$$

**Step 5:** Update weights:

$$W \leftarrow W - \alpha \frac{\partial \mathcal{L}}{\partial W}$$

**Step 6:** Repeat over training dataset

### What Gets Learned

Through training, different dimensions specialize:

- Some dimensions track long-term state: high forget gate
  $$f_t \approx 1$$

- Some dimensions track short-term events: low forget gate
  $$f_t \approx 0$$

- Some dimensions respond to specific input patterns
- Specialization emerges automatically from gradient descent

---

## Key Insights

### 1. Architectural Roles Are Fixed
The LSTM equations **hardcode** what each matrix does:

- Forget gate must multiply with previous cell state (architectural constraint):
  $$W_f \text{ produces } f_t \text{ which multiplies } c_{t-1}$$

- Input gate must multiply with cell candidate (architectural constraint):
  $$W_i \text{ produces } i_t \text{ which multiplies } \tilde{c}_t$$

- Output gate must multiply with cell state (architectural constraint):
  $$W_o \text{ produces } o_t \text{ which multiplies } \tanh(c_t)$$

You cannot swap their roles - the role comes from their position in the equations.

### 2. Learning Determines Policies
Training determines:
- WHEN to forget (not THAT forget gate controls forgetting)
- WHAT to add (not THAT input gate controls addition)
- WHEN to expose (not THAT output gate controls output)

### 3. Gates Can Learn Opposite Behaviors

Forget gate can learn to NOT forget:

$$f_t \to 1$$

Input gate can learn to block input:

$$i_t \to 0$$

Behavior depends on what minimizes loss.

### 4. Memory Is Compressed, Not Stored
- 48 dimensions don't store 200 frames
- They encode **patterns** extracted from those frames
- Lossy compression optimized by training

### 5. Cell State Can Grow
Cell state can increase despite forget gate being bounded:

$$f_t \leq 1$$

$$c_t = \underbrace{f_t \odot c_{t-1}}_{\leq c_{t-1}} + \underbrace{i_t \odot \tilde{c}_t}_{\text{can be large}}$$

The input term allows accumulation.

---

## Parameter Count

For `hidden_size=h` and `input_size=d`:

**Weight matrices:**

$$4 \times h \times (h + d)$$

**Bias vectors:**

$$4 \times h$$

**Total:**

$$4h(h + d + 1)$$

**Example:** For $h=48$ and $d=96$:

Weight matrices:

$$4 \times 48 \times 144 = 27,648$$

Biases:

$$4 \times 48 = 192$$

**Total: 27,840 parameters**

---

## Practical Considerations

### Choosing Hidden Size

Larger hidden size:

$$h \uparrow \implies \text{more memory capacity, harder to train}$$

Smaller hidden size:

$$h \downarrow \implies \text{less memory capacity, faster training}$$

Typical values: 32-512 depending on task complexity

### Sequence Length

Training sequence length determines maximum learnable dependency.

Longer sequences:

$$T \uparrow \implies \text{better long-term learning, more computation}$$

Typical values: 50-500 timesteps

### When to Use LSTM
- Sequential data with long-term dependencies
- Time series prediction
- Natural language processing
- Control tasks requiring temporal memory
- NOT for: Simple feedforward tasks (overkill)
- NOT for: When computational cost is prohibitive

---

## Summary

**LSTM = Regular RNN + Gating Mechanism + Separate Cell State**

**Core Innovation:**
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

The **addition** (not multiplication) enables gradient flow through time, solving the vanishing gradient problem.

**Three Key Ideas:**

1. Separate internal memory from output:
   $$c_t \text{ (internal)}, \quad h_t \text{ (output)}$$

2. Additive updates to cell state

3. Learned gates control information flow

**Bottom Line:** LSTMs are just four learned weight matrices that produce multipliers, combined with a carefully designed equation structure that prevents vanishing gradients and enables long-term memory.

---

## Regular RNN vs LSTM

**Regular RNN (Vanilla RNN)**

State: Single hidden state $h_t$

Equation:
$$h_t = \tanh(W_h \cdot h_{t-1} + W_x \cdot x_t + b)$$

Problems:
- Vanishing gradients through repeated multiplication
- Cannot learn long-term dependencies (typically 5-10 timesteps max)
- Information decays rapidly

**LSTM**

States: Two separate vectors
- $h_t$: Hidden state (filtered output, exposed externally)
- $c_t$: Cell state (internal memory, long-term storage)

**Key Advantage:** Additive cell state updates prevent vanishing gradients, enabling learning of 100+ timestep dependencies.

**Key Difference:** Regular RNN has no cell state vector, only hidden state. LSTM separates internal memory from exposed output, allowing long-term information storage without interference.
