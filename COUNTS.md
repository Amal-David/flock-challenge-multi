# Ligerito Cross-Level Redundancy & Table Slicing Census

## Target Overview
Investigation into cross-level table, basis vector, and equality factor redundancy across the recursive Ligerito opening (`21.2 ms` total open phase; `14.8 ms` recursive prover).

Ranked Proof Shape (BLAKE3-2^18, $m=32, \text{initial\_k}=6$):
- L0: $N = 2^{25}$, folded witness $f_1 \in \mathbb{F}_{2^{128}}^{2^{19}}$, $2^{20}$ leaves, 64 lanes.
- L1: $2^{18}$ leaves, 8 lanes, $f_2 \in \mathbb{F}_{2^{128}}^{2^{16}}$ ($k_1=3$).
- L2: $2^{16}$ leaves, 8 lanes, $f_3 \in \mathbb{F}_{2^{128}}^{2^{13}}$ ($k_2=3$).
- L3: $2^{14}$ leaves, 8 lanes, $f_4 \in \mathbb{F}_{2^{128}}^{2^{10}}$ ($k_3=3$).
- L4: $2^{12}$ leaves, 8 lanes, $f_5 \in \mathbb{F}_{2^{128}}^{2^{7}}$ ($k_4=3$).
- L5: $2^{10}$ leaves, 8 lanes, $y_r \in \mathbb{F}_{2^{128}}^{16}$ ($k_5=3$).

---

## 1. Subspace Polynomial Evaluation (`eval_sk_at_vks` & `inv_sks_vks`)

### Citations
- `crates/flock-core/src/pcs/ligerito.rs:1764-1784`: `eval_sk_at_vks` polynomial recurrence.
- `crates/flock-core/src/pcs/ligerito.rs:2071-2074`: `inv_sks_vks` per-entry field inversion loop.
- `crates/flock-core/src/pcs/ligerito.rs:3112-3128`: `induce_sumcheck_poly_auto` lazy dense-arm dispatch.
- `crates/flock-core/src/pcs/ligerito.rs:9786-9801`: `recursive_verifier_with_basis_succinct` residual check.

### Per-Proof Counts
On the ranked prover path:
- L0 & L1 take the sparse-NTT / sparse-dual arm (`use_ntt = true`).
- L2 ($\log d = 13$), L3 ($\log d = 10$), L4 ($\log d = 7$) take the dense arm (`induce_sumcheck_poly`):
  - L2 ($\log d = 13$): $\frac{13 \times 14}{2} = 91$ field multiplications, 14 field inversions ($14 \times 140 \approx 1,960$ ops).
  - L3 ($\log d = 10$): $\frac{10 \times 11}{2} = 55$ field multiplications, 11 field inversions ($11 \times 140 \approx 1,540$ ops).
  - L4 ($\log d = 7$): $\frac{7 \times 8}{2} = 28$ field multiplications, 8 field inversions ($8 \times 140 \approx 1,120$ ops).
- Total field multiplications deleted: **174 multiplies**.
- Total field inversions deleted: **33 inversions** ($\approx 4,620$ field ops).
- Total dynamic allocations deleted: **6 heap Vec allocations** (3 `sks_vks` + 3 `inv_sks_vks`).

### Millisecond Ceiling
- 174 multiplies + 4,620 inversion operations = ~4,800 clock cycles on Sapphire Rapids.
- At 3.2 GHz, 4,800 cycles = $1.5 \ \mu\text{s} = \mathbf{0.0015\text{ ms}}$.
- Ceiling if 100% deleted: **0.0015 ms** (+0.01 bips).

---

## 2. Equality Tables in OOD & Induce (`build_eq_table`)

### Citations
- `crates/flock-core/src/pcs/ligerito.rs:7741-7742`: `introduce_new_ood_factorized` `eq_lo` / `eq_hi`.
- `crates/flock-core/src/pcs/ligerito.rs:2681, 2685`: `SparseDualL0::new` `lane_weights` & `alpha_pows`.
- `crates/flock-core/src/pcs/ligerito.rs:2056, 2065`: `induce_sumcheck_poly` `eq` & `alpha_pows`.

### Per-Proof Counts & Cross-Level Independence
- **OOD Equality Tables**:
  - L1 ($z \in \mathbb{F}^{19}$): $2^{11} + 2^7 = 2,176$ entries (2,174 mults).
  - L2 ($z \in \mathbb{F}^{16}$): $2^{11} + 2^4 = 2,064$ entries (2,062 mults).
  - L3 ($z \in \mathbb{F}^{13}$): $2^{11} + 2^1 = 2,050$ entries (2,048 mults).
  - L4 ($z \in \mathbb{F}^{10}$): $2^9 + 2^0 = 513$ entries (511 mults).
  - L5 ($z \in \mathbb{F}^{7}$): $2^6 + 2^0 = 65$ entries (63 mults).
  - Total entries: 6,868 entries, ~6,858 field multiplies.
  - **Cross-level restriction potential: 0 entries**. Each level samples a fresh random point $z_{i} \leftarrow \text{Challenger}$ drawn *after* observing $\text{root}_i$; $z_{i+1}$ is statistically independent of $z_i$ and cannot be restricted or sliced.
- **Induce / Sparse Dual Equality Tables**:
  - `lane_weights` across 5 levels: $64 + 8 + 8 + 8 + 8 = 96$ entries (91 mults).
  - `alpha_pows` across 5 levels: $256 + 128 + 128 + 64 + 64 = 640$ entries (635 mults).
  - Total: 736 entries, 726 multiplies.
  - **Cross-level restriction potential: 0 entries** (challenges $\alpha_i$ and fold weights are drawn per level).

### Millisecond Ceiling
- Total table generation cost across all 5 levels: ~7,584 multiplies $\approx 2.4 \ \mu\text{s} = \mathbf{0.0024\text{ ms}}$.
- Entire OOD phase measured: **0.22 ms** (dominated by MLE evaluation against $f$).
- Entire induce phase measured: **0.27 ms** across all 5 levels.
- Ceiling if 100% of all table generation is deleted: **0.0024 ms** (+0.015 bips).

---

## 3. Nested Recursive Commits (`ligero_commit`)

### Citations
- `crates/flock-core/src/pcs/commit.rs:728-755`: `fused_encode_leaves_subtree`.
- `crates/flock-core/src/pcs/ligerito.rs:3691-3760`: `ligero_commit`.
- `crates/flock-core/src/pcs/ligerito.rs:9210`: Level commit dispatch in recursive prover.

### Per-Proof Measured Breakdown (Warm c7i trace)
- L1 commit ($2^{18}$ leaves, 32 MB): 3.75 ms (overlapped inside direct fold 8 / M6).
- L2 commit ($2^{16}$ leaves, 8 MB): **0.66 ms**.
- L3 commit ($2^{14}$ leaves, 2 MB): **0.20 ms**.
- L4 commit ($2^{12}$ leaves, 0.5 MB): **0.11 ms**.
- L5 commit ($2^{10}$ leaves, 0.1 MB): **0.07 ms**.
- Total recursive commit duration (L2–L5): **1.04 ms** (measured at **1.12 ms** in full trace).
- **Cross-level carrying/slicing potential: 0 ms**. Each level commits to the folded witness $f_k(x) = f_{k-1}(2x) + r \cdot (f_{k-1}(2x+1) + f_{k-1}(2x))$ under Fiat-Shamir fold challenges $r$; the codeword is an algebraic transform of the new random linear combination, not a sub-codeword.

---

## 4. Total Census Summary & Near-Miss Determination

| Component | Per-Proof Target Work | Measured Duration | Removable Headroom (Ceiling) |
|---|---|---|---|
| `sks_vks` & `inv_sks_vks` tables | 174 mults, 33 invs, 6 allocs | 0.0015 ms | **0.0015 ms** |
| OOD & Induce `build_eq_table` | 7,584 mults | 0.0024 ms | **0.0024 ms** |
| All 5 Induce Basis steps combined | 5 levels (L0..L4) | 0.27 ms | **0.27 ms** |
| All 5 OOD steps combined | 5 evaluations (L1..L5) | 0.22 ms | **0.22 ms** |
| All 5 Introduce + Glue steps combined | 5 introductions | 0.19 ms | **0.19 ms** |
| Sumcheck recursive folds | 15 AVX-512 fused folds | 2.03 ms | N/A (protocol dependency) |
| Recursive Commits (L2..L5) | 4 NTT+Merkle trees | 1.12 ms | N/A (algebraic dependency) |

### Stopping Condition Rule Check
- Target mechanism ceiling (cross-level table / basis slicing): **0.0015 ms – 0.0039 ms** ($\ll 0.8\text{ ms}$).
- Aggregate ceiling if 100% of ALL non-fold/non-commit Ligerito opening work is deleted: $0.27 + 0.22 + 0.19 = \mathbf{0.68\text{ ms}} < 0.80\text{ ms}$.
- **Result**: Quantified Near-Miss. The 100% deletion ceiling of the candidate mechanism is well below the mandatory 0.8 ms stopping threshold.
