Overview
--------

**Key Concepts:**
- **Environment**: The world the agent interacts with (e.g., a game, robot simulator)
- **State** ($s_t$): Current situation/observation at time $t$
- **Action** ($a_t$): What the agent does in state $s_t$
- **Reward** ($r_t$): Immediate feedback from environment after taking action $a_t$

PPO
---

Proximal Policy Optimization is a RL algorithm that balance performance with training stability.

### Two Networks (Actor-Critic)

PPO trains two neural networks simultaneously:

**1. Policy Network (Actor)** - $\pi_\theta(a|s)$
- Input: state $s_t$ → Output: action probabilities
- Purpose: Decides what action to take
- Loss: $L^{CLIP}$ (clipped surrogate objective)

**2. Value Network (Critic)** - $V_\phi(s)$
- Input: state $s_t$ → Output: estimated future reward
- Purpose: Evaluates how good states are (used to compute advantages)
- Loss: $L^{VF}$ (mean squared error)

**Training Process:**
1. **Collect experiences**: Agent interacts with environment using current policy
   - Collect: states $[s_0, s_1, ..., s_T]$, actions $[a_0, a_1, ...]$, rewards $[r_0, r_1, ...]$

2. **Compute advantages**:
   - Forward pass value network on all states: get all value estimates
   - Calculate TD residuals for each timestep
   - Calculate GAE to get all advantages
   - Detach advantages from gradient computation (treat as constants)

3. **Update both networks** (multiple epochs on same batch):
   - **Policy network**: 
     - Forward pass: compute $\pi_\theta(a_t|s_t)$ for all states
     - Compute $L^{CLIP}$ using frozen advantages $\hat{A}_t$
     - Backprop and update $\theta$
   - **Value network**:
     - Forward pass: compute $V_\phi(s_t)$ for all states
     - Compute $L^{VF}$ comparing to targets $V^{target}_t$
     - Backprop and update $\phi$
   - (Often combined into single $L^{TOTAL}$ and updated together)

4. **Repeat**: Collect new experiences with updated policy

### Loss Function

**Policy Loss**

$$
L^{CLIP}(\theta) = -\hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ is the probability ratio
- $\hat{A}_t$ is the estimated advantage at time t
- $\epsilon$ is the clipping parameter (typically 0.2)

**Why clipping?** The clip operation prevents the policy from changing too much in one update:
- If $\hat{A}_t > 0$ (good action): penalize if $r_t > 1+\epsilon$ (policy changed too much)
- If $\hat{A}_t < 0$ (bad action): penalize if $r_t < 1-\epsilon$ (policy changed too much)
- This keeps the new policy close to the old one, ensuring stable training

**Value Loss (MSE)**

$$
L^{VF}(\phi) = \hat{\mathbb{E}}_t[(V_\phi(s_t) - V^{target}_t)^2]
$$


**Total Loss**

$$
L^{TOTAL} = L^{CLIP}(\theta) + c_1 L^{VF}(\phi) - c_2 S[\pi_\theta]
$$

where:
- $L^{VF}(\phi)$ is the value loss: mean squared error between predicted and target values
- $S[\pi_\theta]$ is the entropy bonus (encourages exploration)
- $c_1, c_2$ are coefficients (typically $c_1 = 0.5$, $c_2 = 0.01$)

**Gradient Derivation:**

The gradient of the clipped loss is:

$$
\nabla_\theta L^{CLIP}(\theta) = -\hat{\mathbb{E}}_t \left[ \nabla_\theta \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]
$$

For each timestep, the gradient depends on which term is smaller:

**Case 1**: If $r_t(\theta) \hat{A}_t < \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t$ (unclipped):

$$
\nabla_\theta L_t = -\hat{A}_t \nabla_\theta r_t(\theta) = -\hat{A}_t \nabla_\theta \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} = -\hat{A}_t \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

**Case 2**: If clipped (when $r_t(\theta) > 1+\epsilon$ and $\hat{A}_t > 0$, or $r_t(\theta) < 1-\epsilon$ and $\hat{A}_t < 0$):

$$
\nabla_\theta L_t = 0
$$

During gradient computation, advantages are treated as constant coefficients (no gradients flow through them).

### How is Advantage calculated

The advantage function measures how much better an action is compared to the average action at a given state. By definition:

$$
A_t = Q(s_t, a_t) - V(s_t)
$$

where $Q(s_t, a_t)$ is the action-value (how good is action $a_t$ in state $s_t$) and $V(s_t)$ is the state-value (average value of state $s_t$).

Since we don't know the true $Q$ and $V$ functions, we use **Generalized Advantage Estimation (GAE)**:

$$
\hat{A}_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \cdots
$$

where the TD residual $\delta_t$ is:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

Parameters:
- $\gamma$ (gamma): discount factor for future rewards (typically 0.99)
- $\lambda$ (lambda): GAE parameter that controls bias-variance tradeoff (typically 0.95)
- $V(s_t)$: value function estimate at state $s_t$
- $r_t$: reward at time $t$

### How is V(s) Calculated

The value function $V(s_t)$ is computed by the **value network** (critic):

**Training Target:**

The value network learns by minimizing the difference between its prediction and the actual observed return:

$$
V^{target}_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots + \gamma^{T-t} r_{T-1}
$$

This is the **actual total discounted reward** observed from state $s_t$ to the end of the episode.

**Training:**
- Collect trajectory and observe all rewards
- Compute targets: $V^{target}_t$ for each state (sum of future rewards)
- Update network to minimize: $L^{VF} = (V_\phi(s_t) - V^{target}_t)^2$
- Over time, the network learns to predict future rewards accurately

**Usage:**
- Once trained, just forward pass: $V(s_t) = V_\phi(s_t)$
- Used to compute TD residuals and advantages
- Continuously updated during PPO training
