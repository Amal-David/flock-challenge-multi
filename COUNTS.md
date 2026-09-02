# Zerocheck Round 1 AB Plane Accumulation: Source Counts and Near-Miss Ceiling

## Target Symbol and Source Citations

- **Tree**: `lane/zc-r1-plane` (base commit `c75aeceb34f8f33ff49c06a293b05dd7e00d463b`)
- **Target Kernel**: `convert_ab_nomul_x86_gfni_direct<FIRST, N, FIRST_WRITE>`
- **Source Files**:
  - `crates/flock-core/src/zerocheck/univariate_skip_optimized.rs:3339-3611` (`process_one_x_hi_ab_only`)
  - `crates/flock-core/src/zerocheck/univariate_skip_optimized.rs:3742-3752` (`eq_fold_state` parameters: `n_lo = 12`, `bank_bits = 7`, `hi_size = 128`, `big_lo_size = 4096`)
  - `crates/flock-core/src/zerocheck/univariate_skip_optimized/kernels.rs:610-645` (`convert_ab_nomul_gfni_direct`)
  - `crates/flock-core/src/zerocheck/univariate_skip_optimized/kernels/x86_64.rs:1930-2007` (`convert_ab_nomul_x86_gfni_direct`)

---

## Per-Proof Invocation and Uop Census

### 1. Proof Geometry & Invocations
- **Bands (`hi_size`)**: 128 bands
- **Windows per band (`big_lo_size`)**: 4,096 windows (`1 << n_lo = 1 << 12`)
- **Total Window Invocations**: $128 \times 4096 = 524,288$ windows per proof
- **Window Distribution**:
  - Parity 0 (even windows): $(first\_b\_med, n\_b\_med) = (2, 16)$ $\rightarrow 14$ live rows ($262,144$ windows)
  - Parity 1 (odd windows): $(first\_b\_med, n\_b\_med) = (0, 15)$ $\rightarrow 15$ live rows ($262,144$ windows)
- **Total Live Rows**: $262,144 \times 14 + 262,144 \times 15 = 7,602,176$ rows

---

### 2. Instruction & Uop Breakdown per Proof

| Operation | Per $(2, 16)$ Window | Per $(0, 15)$ Window | Per-Proof Total | Port Binding (Sapphire Rapids) |
|---|---|---|---|---|
| **Input Row Loads** (`vmovdqu64`) | 14 | 15 | 7,602,176 | Port 2/3 (Load) |
| **Plane Accumulator Loads** (`vpternlogq` / `vmovdqu64`) | 15.5 (31/32 of 16) | 15.5 (31/32 of 16) | 8,126,464 | Port 2/3 (Load) |
| **Plane Accumulator Stores** (`vmovdqu64`) | 16 | 16 | 8,388,608 | Port 4/7/8 (Store) |
| **GFNI Matrix Mult** (`vgf2p8affineqb` + `{1to8}`) | 224 ($14 \times 16$) | 240 ($15 \times 16$) | 121,634,816 | Port 0 (GFNI) |
| **Accumulator XOR / Ternary** (`vpternlogq` / `vpxord`) | 112 ($7 \times 16$) | 128 ($7\times 16 + 16$) | 62,914,560 | Port 0/5 (ALU/Logic) |

---

### 3. Removable Uop Census (Target Deletion)

The targeted candidate deletion is the elimination of intermediate plane store+load round-trips per window:
- **Plane Loads Deleted**: $8,126,464$ 512-bit vector loads
- **Plane Stores Deleted**: $8,388,608$ 512-bit vector stores
- **Total Memory Uops Targeted**: $16,515,072$ uops ($\approx 16.52\text{ M uops}$)
- **Mathematical GFNI Transformations (Mandatory)**: $121.63\text{ M}$ operations (cannot be deleted without breaking algebraic correctness).

---

### 4. Time Ceiling Estimation at Saturated Throughput

On AWS `c7i.4xlarge` (Intel Xeon 8488C Sapphire Rapids, 8 cores $\times$ 2 SMT @ $\approx 3.0\text{ GHz}$):
- Saturated port throughput aggregate: $\approx 24\text{ M uops/ms}$
- **Theoretical Deletion Ceiling**:
  $$\text{Ceiling} = \frac{16,515,072\text{ uops}}{24,000,000\text{ uops/ms}} = \mathbf{0.688\text{ ms}} \approx \mathbf{0.69 - 0.70\text{ ms}}$$
- **Timed Proof Baseline (Crown `c75aeceb`)**: $160.5\text{ ms}$ (median proof)
- **Promotion Threshold (+100 bips = +1.00%)**: Requires $\mathbf{\ge 1.605\text{ ms}}$ of proof time reduction ($> 1,649,903.13\text{ comp/s}$).
- **Maximum Achievable Gain**:
  $$\Delta = \frac{0.688\text{ ms}}{160.5\text{ ms}} \times 10,000\text{ bips} = \mathbf{+42.9\text{ bips}} \quad (\ll 100\text{ bips})$$

---

### 5. Conclusion & Stop Decision

Because the theoretical upper-bound deletion ceiling for eliminating all plane store/load round-trips is **$0.69\text{ ms} < 1.50\text{ ms}$**, the mechanism cannot meet the $+100\text{ bip}$ promotion requirement. 
In accordance with Output Contract Step 1:
> *"if the ceiling is < 1.5 ms, write the near-miss and STOP."*

This candidate is classified as a verified **Near-Miss / Refutation**.
