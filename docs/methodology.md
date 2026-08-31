# Scientific Methodology & Mathematical Foundations

## 1. Physical Simulation Engine

The simulation engine synthesizes continuous daily records (1,825 days / 5 years) across Northern and Southern hemisphere climate and human behavior dynamics.

### 1.1 Precipitation & Antecedent Moisture
- **Rainfall Occurrence**: Two-state Markov chain $P(\text{Wet}_t \mid \text{State}_{t-1})$.
- **Rainfall Intensity**: Gamma-distributed precipitation $I \sim \text{Gamma}(\alpha, \beta)$ parameterized by seasonal sea-surface temperature anomalies.
- **Antecedent Moisture Condition (AMC)**:
  $$\text{AMC}_t = \sum_{k=1}^5 \gamma^k \cdot R_{t-k}, \quad \gamma = 0.85$$
- **Consecutive Dry Days (CDD)**: Counts dry intervals ($R_t = 0$), resetting to zero on any wet day ($R_t > 0$).

### 1.2 Soil Runoff & Contaminant Transport
- **SCS Curve Number Runoff**:
  $$Q = \frac{(P - I_a)^2}{(P - I_a) + S}, \quad S = \frac{25400}{\text{CN}} - 254, \quad I_a = 0.2 S$$
- **Total Suspended Solids (TSS)**: First-flush washoff model dependent on dry day accumulation and peak runoff volume:
  $$\text{TSS}_t = \beta_0 \cdot (1 - e^{-\kappa \cdot \text{CDD}_t}) \cdot Q_t^\alpha$$
- **Nutrient Load Index**: Synergistic interaction between fertilizer wash-off and temperature-driven microbial kinetics:
  $$\text{NLI}_t = \text{TSS}_t \cdot \exp\left(\frac{T_t - 20}{10}\right)$$

---

## 2. Unsupervised Behavioral Clustering (Service B)

Service B identifies 4 consumer archetypes based on 4 engineered behavioral features:

1. $\log(\text{per\_capita\_usage})$: Log-transformed average volume per household occupant.
2. $\text{dry\_day\_spike\_factor}$: Ratio of dry-period consumption to baseline consumption.
3. $\text{efficiency\_penalty\_ratio}$: Fraction of total volume exceeding standard tier allowances.
4. $\text{landscape\_demand\_index}$: Correlation between outdoor temperature and peak irrigation.

### Archetype Profiles:
- **Cluster 0**: *Conservationists (Low Volume)* — High efficiency penalty adherence, minimal dry-day spike.
- **Cluster 1**: *Standard Average Consumers* — Moderate demand tracking seasonal patterns.
- **Cluster 2**: *Outdoor / Landscape Heavy* — High landscape demand index sensitive to heat waves.
- **Cluster 3**: *Heavy Users (High Volume)* — Top decile per-capita consumption, high spike factor.

---

## 3. Supervised WQI Prediction Pipeline (Service A)

### 3.1 Target Definition
The Water Quality Index ($\text{WQI} \in [0, 100]$) aggregates physical and chemical indicators:
$$\text{WQI}_t = 100 - \left( w_1 \cdot \frac{\text{TSS}_t}{\text{TSS}_{\max}} + w_2 \cdot \frac{\text{NLI}_t}{\text{NLI}_{\max}} + w_3 \cdot \text{TempAnomaly}_t \right)$$

### 3.2 Feature Engineering & Interaction Terms
- **Lag Features**: Optimal lags identified by hemisphere ($L=7$ for North, $L=3$ for South).
- **Physical Interactions**:
  - $\text{Heat} \times \text{Nutrient Synergy} = \text{CumulativeHeatIndex} \cdot \text{NutrientLoadIndex}$
  - $\text{Demand} \times \text{Runoff Pressure} = \text{TotalClusterDemand} \cdot \text{DailyRunoffVolume}$
  - $\text{Drought} \times \text{Heat Stress} = \text{ConsecutiveDryDays} \cdot \text{DailyMaxTemp}$

### 3.3 Multi-Stage Selection Pipeline
1. **Screening**: Excludes constant features, zero-variance columns, and explicit leakages.
2. **Stability Selection**: Subsampled Lasso / Random Forest selection preserving features with selection frequency $\pi > 0.60$.
3. **Family Ablation**: Evaluates marginal contribution of policy, cluster, weather, and lag families.

### 3.4 Level-1 Ensemble & Residual AR Correction
- **Weighted Blending**: SLSQP constrained optimization $\min_w \sum (y - \sum w_i \hat{y}_i)^2$ s.t. $\sum w_i = 1, w_i \ge 0$.
- **Stacking Meta-Learner**: Ridge regression fitted on out-of-fold level-0 validation matrix.
- **Autoregressive Residual Corrector**:
  $$e_t = y_t - \hat{y}_t^{\text{ensemble}}, \quad \hat{e}_{t+1} = \sum_{j=1}^p \phi_j e_{t-j+1}$$
  $$\hat{y}_{t+1}^{\text{final}} = \text{clip}\left(\hat{y}_{t+1}^{\text{ensemble}} + \lambda \hat{e}_{t+1}, 0, 100\right)$$
