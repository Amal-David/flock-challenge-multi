# Optimization Proposal: Ligerito Cross-Level Subspace Polynomial Table Slicing & Near-Miss Investigation

## 1. Executive Summary & Outcome Contract
- **Classification**: **Quantified Near-Miss / Truthful Negative**.
- **Scope**: Ligerito recursive opening cross-level table and basis redundancy.
- **Target Mechanism**: Memoized global standard subspace-polynomial evaluation (`sks_vks`) and inversion (`inv_sks_vks`) table slicing across recursive Ligerito levels (L2–L4 prover and succinct verifier).
- **Counted Deleted Operations**: 174 field multiplications, 33 field inversions (~4,620 arithmetic ops), 6 heap allocations per proof.
- **Estimated Headroom**: **0.0015 ms** (+0.01 bips).
- **Stopping Rule Ceiling**: The entire non-fold/non-commit Ligerito opening budget (induces + OOD + glue) is **0.68 ms** (< 0.80 ms threshold). The candidate mechanism ceiling is **0.0015 ms**, far below the +100 bip (+1.00% = ~1.6 ms) promotion bar and below the 0.8 ms threshold.

---

## 2. Mechanism & Architectural Analysis

### A. Subspace Polynomial Invariance (`sks_vks` & `inv_sks_vks`)
In LCH additive NTT and Ligerito basis induction, the subspace polynomial sequence is defined by the recurrence:
$$s_0(x) = x, \quad s_{i+1}(x) = s_i(x)^2 + s_i(v_i) \cdot s_i(x)$$
Over the canonical standard basis $v_i = 1 \ll i$, $s_k(v_k)$ is a deterministic constant depending only on index $k \in [0, 64]$, invariant to the total dimension $\log n$.

**Incumbent Behavior (`c75aeceb`)**:
- In `induce_sumcheck_poly_auto` (`ligerito.rs:3116`), whenever the dense induce arm is taken (levels L2, L3, L4), `eval_sk_at_vks(\log d)` re-evaluates the $O(\log d^2)$ triangular recurrence from scratch.
- In `induce_sumcheck_poly` (`ligerito.rs:2071`), `inv_sks_vks` allocates a new `Vec<F128>` and performs $\log d + 1$ Fermat inversions per level.
- In `recursive_verifier_with_basis_succinct` (`ligerito.rs:9789`), the succinct verifier reconstructs `eval_sk_at_vks` across all recursive levels.

**Proposed Mechanism**:
- `STANDARD_SKS_VKS_TABLES`: Precomputes the static tables for dimension 64 once at initialization.
- Prover and verifier levels take zero-allocation sub-slices `&sks_vks[..=\log d]` and `&inv_sks_vks[..=\log d]`.
- Rollback: Gated behind `FLOCK_NO_OPEN_SKS_SLICE=1`.

### B. Cross-Level Equality Tables & Codeword Redundancy Audit
- **OOD Equality Tables** (`ligerito.rs:7741`): At each level $i$, the evaluation point $z_i \in \mathbb{F}^{n_i}$ is a fresh pseudo-random challenge sampled from the transcript *after* observing $\text{root}_i$. Because $z_{i+1}$ is statistically independent of $z_i$, cross-level tensor slicing of $eq(z, \cdot)$ is algebraically impossible. Total OOD phase takes **0.22 ms**.
- **Induce Equality Tables** (`ligerito.rs:2681, 2685`): Per-level query weights $\alpha_i$ and fold weights are similarly fresh challenges. Total table generation takes **0.0024 ms** across all levels.
- **Nested `ligero_commit` Levels** (`commit.rs:728`): Commits at $2^{18}, 2^{16}, 2^{14}, 2^{12}, 2^{10}$ leaves commit to the folded witness $f_k$, which is a non-trivial random linear combination of previous coefficients under fold challenges $r$. The codewords cannot be restricted or sliced from level $k-1$. Total recursive commit cost is **1.12 ms**.

---

## 3. Counted Deletion with Exact Citations

| Location | Incumbent Operation | Deletion / Slicing Optimization | Operations Deleted |
|---|---|---|---|
| `crates/flock-core/src/pcs/ligerito.rs:1764` | `eval_sk_at_vks` per level | Sub-slice `&STANDARD_SKS_VKS[..=log_n]` | 174 field mults |
| `crates/flock-core/src/pcs/ligerito.rs:2071` | `inv_sks_vks` inversion loop | Sub-slice `&STANDARD_INV_SKS_VKS[..=log_n]` | 33 field inversions |
| `crates/flock-core/src/pcs/ligerito.rs:3116` | Dynamic allocation in `auto` | Direct static reference | 3 heap allocations |
| `crates/flock-core/src/pcs/ligerito.rs:9789` | Verifier residual `eval_sk_at_vks` | Static sub-slice | 5 redundant evals |

---

## 4. Headroom, Latency, and Score Ceiling

- **Total Clock Cycles Saved**: ~4,800 cycles on Sapphire Rapids.
- **Estimated Net Latency Reduction**: $\mathbf{0.0015\text{ ms}}$ ($1.5\ \mu\text{s}$).
- **Score Impact**: $\mathbf{+0.01\text{ bips}}$ ($+0.0001\%$).
- **Theoretical Ceiling (100% deletion of all table generation)**: $\mathbf{0.0039\text{ ms}}$ (+0.024 bips).
- **Theoretical Ceiling (100% deletion of all Induce + OOD + Glue)**: $\mathbf{0.68\text{ ms}}$ (+4.2 bips).

Because the promotion threshold requires $\ge +100\text{ bips}$ ($\approx 1.6\text{ ms}$) and the near-miss ceiling threshold is $0.8\text{ ms}$, this candidate is definitively classified as a near-miss.

---

## 5. Correctness & Byte-Identity Argument
1. **Mathematical Invariance**: Standard basis vectors $v_i = 1 \ll i$ are fixed constants. By induction on the recurrence $s_{i+1}(x) = s_i(x)^2 + s_i(v_i) s_i(x)$, $s_k(v_k)$ is invariant to any truncation or extension of the basis dimension.
2. **Field Inversion**: Field inversion in $\mathbb{F}_{2^{128}}$ is unique for non-zero elements; the static table matches element-wise `v.inv()`.
3. **Determinism**: The Fiat-Shamir transcript is byte-identical because all observed messages, challenge points, commitments, and proof structures are strictly unchanged.
4. **Unit Test Verification**: `pcs::ligerito::tests::open_sks_slice_matches_uncached_and_rollback` verifies bit-identical equality of `sks_vks`, `inv_sks_vks`, `induce_sumcheck_poly` basis output, and `enforced_sum` against the uncached oracle across dimensions $0..=32$.

---

## 6. Rollback Mechanism & Verification
- **Environment Variable**: `FLOCK_NO_OPEN_SKS_SLICE=1`
- When set, `open_sks_slice_enabled()` evaluates to `false`, executing the exact incumbent dynamic calculation path byte-for-byte.

---

## 7. Native Validation Output (AWS c7i.4xlarge)

- **Target Hardware**: AWS c7i.4xlarge (Intel Xeon Platinum 8488C Sapphire Rapids, AVX-512 / VPCLMULQDQ / GFNI).
- **Branch**: `lane/open-ligerito`
- **Commit SHA**: `56363a06`
- **Command**: `/Users/amal/bin/validate_on_aws.sh lane/open-ligerito eval_sk_at_vks`
- **Results**:
  - Scope Check: `crates/flock-core/src/pcs/ligerito.rs` (Clean, scope OK).
  - Native Build (`--profile challenge`, `-C target-cpu=native`): Clean build, 0 errors.
  - Native Unit Tests (`flock-core` on AVX-512): **469 passed, 0 failed** (includes new rollback & slicing parity test).
  - Assembly Frame & Instruction Size: Clean, zero-frame inline dispatch.

---

## 8. Residual Risks & Campaign Conclusion
- **Risks**: None. Changes are fully contained behind the verified rollback flag and maintain bit-identical transcript output.
- **Campaign Recommendation**: As established by the source-counted census, cross-level table/basis slicing in Ligerito yields at most 0.0015 ms. The entire remaining non-fold/non-commit Ligerito opening work is only 0.68 ms. Micro-optimizations on this phase cannot deliver the +100 bip promotion requirement.
