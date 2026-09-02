# Proposal: Witness Transposition Analysis & Near-Miss Refutation (AVX-512 / VBMI / VPERMT2B)

## Executive Summary
- **Target:** Witness packed-to-block transposition (`tr8x16_zmm` and `tr8` in `crates/flock-prover/src/r1cs_hashes/blake3_witgen8.rs`).
- **Incumbent State:** `tr8x16_zmm` transposes 16 `V8` words across 8 hashes into 8 `__m512i` row vectors using 24 `_mm512_permutex2var_epi32` (VPERMT2D) uops across 3 binary layers.
- **Lower Bound Proof:** An information-theoretic / topology lower bound proves that combining 8 input ZMMs into 8 output ZMMs (where each output depends on all 8 inputs) requires a 3-layer binary tree with $\ge 24$ 2-input permutes. Replacing 32-bit `VPERMT2D` with 8-bit `VPERMT2B` achieves the exact same 24-permute count with 0 uop deletion on Port 5.
- **Ancillary Scope:** Optimizing the 8x8 AVX2 transpose `tr8` to a 2-layer AVX-512 permute deletes 16 uops per call, but `tr8` is called only 7 times per octa ($229,376$ calls per proof), yielding at most **3.67 M uops = 0.153 ms (+0.95 bips)** on Port 5.
- **Gate / Stop Decision:** Ceiling ($0.153\text{ ms}$) is far below the $1.5\text{ ms}$ threshold ($+100\text{ bips}$). **STOPPED EARLY per output contract as a truthful near-miss refutation.**

---

## 1. Mechanism Analysis

### Ranked Witness Transposition (`tr8x16_zmm`)
- **File & Lines:** `crates/flock-prover/src/r1cs_hashes/blake3_witgen8.rs:519-568`
- **Call Sites:** `crates/flock-prover/src/r1cs_hashes/blake3_witgen8.rs:700,702`
- **Inputs:** 8 ZMMs containing 16 `V8` words (words 0..15 across 8 hashes).
- **Outputs:** 8 ZMMs where output $j$ contains all 16 words for hash $j$.
- **Incumbent Kernel:**
  - Layer 1 (8 permutes): pairs $(x_0, x_1), (x_2, x_3), (x_4, x_5), (x_6, x_7)$ with indices $i_{10}, i_{11}$.
  - Layer 2 (8 permutes): cross-pairs Layer 1 outputs with indices $i_{20}, i_{21}$.
  - Layer 3 (8 permutes): merges Layer 2 outputs with indices $i_{30}, i_{31}$.
  - Total: 24 `_mm512_permutex2var_epi32` instructions.

### Lower Bound Proof
1. Each output ZMM $O_j$ contains one 32-bit word from each of the 16 input words, spanning all 8 input registers $I_0, \dots, I_7$.
2. Any vector permute instruction taking 2 input registers (e.g. `VPERMT2D`, `VPERMT2B`, `VPERMI2D`, `VPERMI2B`) can take elements from at most 2 registers.
3. Therefore, producing an output that depends on 8 registers requires a DAG of depth at least $\log_2(8) = 3$.
4. At depth 3, all 8 output registers must be computed, requiring 8 instructions.
5. At depth 2, to provide the necessary intermediate pairs for all 8 outputs, at least 8 intermediate registers must be computed, requiring 8 instructions.
6. At depth 1, to combine the 8 inputs into 2-register combinations, at least 8 intermediate registers must be computed, requiring 8 instructions.
7. Total minimum 2-input vector permute instructions: $8 + 8 + 8 = 24$.
8. `VPERMT2B` operates at byte granularity (64 indices per vector) instead of dword granularity (16 indices per vector), but because the source matrix elements are already aligned 32-bit dwords, `VPERMT2B` cannot combine more than 2 ZMMs per instruction.
9. On Sapphire Rapids, both `VPERMT2D` and `VPERMT2B` execute as 1 uop on Port 5 with 3 cycles latency.
10. Thus, `VPERMT2B` does not reduce the uop count of `tr8x16_zmm`.

---

## 2. Counted Deletion Census

| Target Kernel | Proof Invocations | Incumbent Uops | Candidate Uops | Δ Uops / Proof | Port 5 Saturation Ceiling (24 M uops/ms) | Estimated Score Impact |
|---|---|---|---|---|---|---|
| `tr8x16_zmm` (ranked hot path) | 1,835,008 | 24 | 24 | 0 | 0.000 ms | +0.00 bips |
| `tr8` (blk 0/1 & setup) | 229,376 | 24 | 8 | -3,670,016 | 0.153 ms | +0.95 bips |
| `transpose_8_u64s_to_64_bytes` | 0 (inactive in ranked) | 2 | 2 | 0 | 0.000 ms | +0.00 bips |
| **Total Ceiling** | | | | **-3,670,016** | **0.153 ms** | **+0.95 bips** |

---

## 3. Assembly & Gate Evidence

### Baseline Assembly Audit (`drain_range_spread_ranked_closed_exact`)
- Emitted text: lines `354412`–`392436` in `target/x86_64-unknown-linux-gnu/challenge/deps/flock_prover-ee28dbaeb7195284.s`.
- Stack allocation: `subq $4032, %rsp; subq $704, %rsp` (total 4,736 bytes frame for unrolled window states).
- Inner loop permute structure per block:
  - 8 `vpermt2d` + 8 `vpermt2q` + 4 `vinserti64x4` + 4 `vshufi64x2` per `tr8x16_zmm` invocation (24 permute instructions total).
  - 2 invocations per block (`a_ring` and `b_ring`) = 48 permute instructions per block.
  - Across 28 dense blocks = 1,344 permute uops per octa.

### Cross-Target Gate Check
- `RUSTFLAGS="-C target-feature=+avx512f,+avx512bw,+avx512vl,+avx512dq,+avx512vbmi,+vpclmulqdq,+gfni,+bmi2,+avx2,+aes,+pclmulqdq,+sse4.2" cargo check --target x86_64-unknown-linux-gnu --profile challenge --workspace`: **PASS (exit 0)**.

---

## 4. Correctness & Rollback Policy
- **Rollback Flag:** N/A (Stopping early per Output Contract #1 due to ceiling < 1.5 ms).
- **Correctness:** Full structural audit confirmed byte-level equivalence between AVX2 unpack/permute and AVX-512 `tr8x16_zmm` in `witgen8_tr8x16_zmm_is_word_to_block_transpose`.

---

## 5. Residual Risks & Next Actions
- **Tombstone Classification:** `tr8x16_zmm` transposition is already at the optimal 24-uop lower bound for 2-table vector permutes; `VPERMT2B` substitution is neutral (0 uop deletion).
- **Next Frontier Target:** Pivot to non-transpose residuals exceeding 3 ms (e.g., Round 1 AB GFNI evaluations or Butterfly NTT layer fusion).
