# MARIN Framework - Mathematical Formulations

## 1. Co-Evolving Multiplex Network Model

### Definition 1: Multiplex Network
```
G = {G₁, G₂, G₃}
```
where each layer `Gₗ = (V, Eₗ(t))` shares a common node set `V` of `N` agents with time-dependent edges.

### Equation 1: Link Formation Probability
```
fₗ(i, j, t) = γₗ · exp(−λₗ · |bᵢ(t) − bⱼ(t)|) + εₗ
```

**Parameters:**
- `bᵢ(t) ∈ [0, 1]`: Belief state of agent i at time t
- `λₗ > 0`: Layer-specific homophily coefficient
- `γₗ ∈ (0, 1]`: Scaling parameter
- `εₗ ≥ 0`: Baseline connection rate
- Constraint: `γₗ + εₗ ≤ 1`

### Equation 2: Link Dissolution Probability
```
gₗ(i, j, t) = σ(δₗ · Dᵢⱼ(t) + φₗ · ΔTᵢⱼ(t) − θₗ)
```

**Parameters:**
- `σ(·)`: Sigmoid function
- `Dᵢⱼ(t) = Σₜ′≤ₜ η^(t−t′)|bᵢ(t′) − bⱼ(t′)|`: Cumulative disagreement exposure
- `η ∈ (0, 1)`: Temporal discount factor
- `ΔTᵢⱼ(t)`: Time since last interaction
- `δₗ, φₗ > 0`: Sensitivity parameters
- `θₗ`: Dissolution threshold

### Equation 3: Cross-Layer Coupling
```
f̃ₗ(i, j, t) = fₗ(i, j, t) + Σₗ′≠ₗ Cₗₗ′ · Aₗ′ᵢⱼ(t) · fₗ′(i, j, t)
```

**Coupling Matrix C:**
```
C₁₂ = 0.30, C₁₃ = 0.10
C₂₁ = 0.20, C₂₃ = 0.25
C₃₁ = 0.15, C₃₂ = 0.35
```

---

## 2. Agent Belief Dynamics

### Equation 4: Belief Update
```
bᵢ(t+1) = (1 − μ) · bᵢ(t) + μ · [κ · Bᵢ(m, t) + (1 − κ) · Sᵢ(t)]
```

**Parameters:**
- `μ ∈ (0, 1)`: Belief update rate (inertia = 1 − μ)
- `κ ∈ [0, 1]`: Confirmation bias weight

### Equation 5: Bayesian Update Component
```
Bᵢ(m, t) = bᵢ(t)^κ · m / [bᵢ(t)^κ · m + (1 − bᵢ(t))^κ · (1 − m)]
```

**Behavior:**
- `κ = 1`: Rational Bayesian updating
- `κ > 1`: Confirmation bias (overweight belief-consistent messages)

### Equation 6: Social Reinforcement
```
Sᵢ(t) = Σₗ ψₗ · [Σⱼ∈Nₗᵢ(t) bⱼ(t)] / [Σₗ ψₗ · |Nₗᵢ(t)|]
```

**Layer Influence Weights:**
- `ψ₁ = 0.2` (Information Sharing)
- `ψ₂ = 0.3` (Social Reinforcement)
- `ψ₃ = 0.5` (Private Communication)
- Constraint: `Σₗ ψₗ = 1`

---

## 3. Reinforcement Learning Architecture

### Equation 7: GNN Message Passing
```
hᵢ⁽ˡ⁾ = ReLU(W⁽ˡ⁾ · AGG({hⱼ⁽ˡ⁻¹⁾ : j ∈ Nᵢ}) + U⁽ˡ⁾ · hᵢ⁽ˡ⁻¹⁾)
```

**Cross-Layer Attention:**
```
zᵢ = Σₗ αₗ · hᵢ⁽ˡ⁾
αₗ = softmax(vᵀ · tanh(Wₐ · hᵢ⁽ˡ⁾))
```

**Global State:**
```
zᴳ = (1/N) Σᵢ zᵢ ∈ ℝ¹²⁸
```

### Equation 8: Node-wise Q-Values
```
qᵢ = Qφ(zᵢ, zᴳ; θ) ∈ ℝ⁴
```

**Intervention Types:**
1. Truth injection
2. Bridge-node activation
3. Echo chamber disruption
4. Cross-layer amplification

### Equation 9: Reward Function
```
R(t) = w₁ · ΔMR(t) + w₂ · ΔBP(t) + w₃ · ECD(t) − w₄ · IC(t)
```

**Default Weights:**
- `w₁ = 0.4` (Misinformation Reach Reduction)
- `w₂ = 0.2` (Belief Polarization Reduction)
- `w₃ = 0.2` (Echo Chamber Disruption)
- `w₄ = 0.2` (Intervention Cost)

### Equation 10: DDQN Update
```
Q(s, a; θ) ← R + γ · Q(s′, argmax_{a′} Q(s′, a′; θ); θ⁻)
```

**Parameters:**
- `γ = 0.99`: Discount factor
- `τ = 1000`: Target network update frequency

---

## 4. Phase Transition Analysis

### Polarization Order Parameter
```
P = (1/N) Σᵢ |2bᵢ − 1|
```

- `P → 0`: Consensus state
- `P → 1`: Full polarization

### Critical Threshold
```
ω* ≈ 0.15
```

**Hysteresis Width:**
```
Δω ≈ 0.04
```

---

## 5. Evaluation Metrics

### Misinformation Reach Reduction (MRR)
```
MRR = (R_baseline − R_intervention) / R_baseline × 100%
```

### Intervention Efficiency (IE)
```
IE = MRR / Budget
```

### Echo Chamber Density (ECD)
```
ECD = 1 − Q(G)
```
where Q(G) is the network modularity.
