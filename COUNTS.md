# Source-Count Analysis: Witness Transposition (AVX-512 / VBMI / VPERMT2B)

## 1. Target Identification and File:Line Citations

### A. Ranked Dense Projection: `tr8x16_zmm`
- **File:** `crates/flock-prover/src/r1cs_hashes/blake3_witgen8.rs:519-568`
- **Caller:** `project_blocks_ranked_hot_offsets_direct_inline` (`crates/flock-prover/src/r1cs_hashes/blake3_witgen8.rs:700,702`) inside `drain_range_spread_ranked_closed_exact` (`crates/flock-prover/src/r1cs_hashes/blake3_witgen8.rs:1841-1916`).
- **Current Mechanism:** 
  - Transposes 16 `V8` words across 8 hashes (128 `u32` words = 512 bytes = 8 ZMMs) into 8 `__m512i` row vectors.
  - Emits 8 `_mm512_loadu_si512` loads + 24 `_mm512_permutex2var_epi32` (VPERMT2D / VPERMT2Q / VINSERTI64X4 / VSHUFI64X2) instructions across 3 binary permute layers (8 + 8 + 8).
- **Invocations per Proof:**
  - Ranked BLAKE3 proof size: $2^{18}$ hashes / 8 = 32,768 octas.
  - Blocks 2..=29: 28 dense hot blocks per octa.
  - Invocations per block: 2 (`a_ring` and `b_ring`).
  - Total `tr8x16_zmm` calls: $32,768 \times 28 \times 2 = 1,835,008$ calls per proof.
- **Current Uop Count per Proof:**
  - $1,835,008 \times 24 = 44,040,192$ permute/shuffle uops on Port 5.

### B. Theoretical Lower Bound for 16x8 `u32` Transpose with 2-Input Permutes
- Input: 8 ZMMs (512 bytes). Output: 8 ZMMs (512 bytes).
- Each output ZMM `rows[j]` requires 16 words distributed across all 8 input ZMMs (1 word per input word).
- Any 2-input vector permute (`_mm512_permutex2var_epi8` / `VPERMT2B` or `_mm512_permutex2var_epi32` / `VPERMT2D`) combines at most 2 input ZMMs into 1 output ZMM.
- A binary tree spanning 8 inputs requires depth $\lceil\log_2(8)\rceil = 3$ layers.
- To produce 8 distinct output ZMMs:
  - Layer 1 must generate 8 intermediate ZMMs (8 permutes).
  - Layer 2 must generate 8 intermediate ZMMs (8 permutes).
  - Layer 3 must generate 8 final ZMMs (8 permutes).
  - Total minimum 2-input permutes: $8 + 8 + 8 = 24$.
- Substituting `VPERMT2B` (`_mm512_permutex2var_epi8`) for `VPERMT2D` (`_mm512_permutex2var_epi32`) changes index granularity from 32-bit dwords to 8-bit bytes, but both instructions execute as 1 uop on Port 5 on Intel Sapphire Rapids with 3-cycle latency.
- **Net uops deleted for `tr8x16_zmm`:** 0 uops.

---

### C. Ancillary AVX2 Transpose: `tr8`
- **File:** `crates/flock-prover/src/r1cs_hashes/blake3_witgen8.rs:465-495`
- **Callers:**
  - `tr8_chunk` (`crates/flock-prover/src/r1cs_hashes/blake3_witgen8.rs:1526-1536`) called for `blk == 0` (2 calls) and `blk == 1` (2 calls) in `drain_range_spread_ranked_closed_exact` (`crates/flock-prover/src/r1cs_hashes/blake3_witgen8.rs:1848,1849,1864,1865`): 4 calls / octa.
  - Setup compression in `witgen8_ring` (`crates/flock-prover/src/r1cs_hashes/blake3_witgen8.rs:2522,2537,2547`): 3 calls / octa (`cv`, `m_lo`, `m_hi`).
  - Total `tr8` calls per octa: $4 + 3 = 7$ calls.
  - Total `tr8` calls per proof: $7 \times 32,768 = 229,376$ calls.
- **Current Mechanism:**
  - 8 `_mm256_unpacklo/hi_epi32` + 8 `_mm256_unpacklo/hi_epi64` + 8 `_mm256_permute2x128_si256` = 24 uops.
- **Optimized AVX-512 VPERMT2D Mechanism:**
  - 4 `_mm512_loadu_si512` + 4 Layer-1 `_mm512_permutex2var_epi32` + 4 Layer-2 `_mm512_permutex2var_epi32` = 8 permute uops.
  - Uop savings per call: $24 - 8 = 16$ uops.
- **Net uops deleted across all `tr8` calls:**
  - $229,376 \text{ calls} \times 16 \text{ uops} = 3,670,016 \text{ uops } (3.67 \text{ M uops})$.

---

### D. Lincheck and Bits Transpose Checks
- `transpose_8_u64s_to_64_bytes_gfni` in `crates/flock-core/src/bits.rs:79-88`:
  - Already optimized to 1 `_mm512_permutexvar_epi8` (VPERMB) + 1 `_mm512_gf2p8affine_epi64_epi8` (GFNI).
  - Not invoked on ranked witness path (modern prover operates directly on block-major packed witness via `partial_fold_packed_z_block_major`).
- Lincheck gather-transpose in `crates/flock-core/src/lincheck.rs:794-810` and `crates/flock-core/src/lincheck/kernels/x86_64.rs:186-196`:
  - Already fused with `VPERMT2B` (`_mm512_permutex2var_epi8(z0, f_lo, z1)`) in a prior epoch (`cff868b`).

---

## 2. Total Deletion and Ceiling Estimation

| Target / Kernel | Invocations / Proof | Incumbent Uops | Optimized Uops | Δ Uops / Proof | Saturated Port | Deletion Ceiling (ms @ 24M uops/ms) |
|---|---|---|---|---|---|---|
| `tr8x16_zmm` (ranked hot path) | 1,835,008 | 24 | 24 (min bound) | 0 | Port 5 | 0.000 ms |
| `tr8` (static blk 0/1 + setup) | 229,376 | 24 | 8 | -3,670,016 | Port 5 | 0.153 ms |
| `transpose_8_u64s_to_64_bytes` | 0 (inactive in ranked) | 2 | 2 | 0 | Port 5 | 0.000 ms |
| **Total Deletion Ceiling** | | | | **-3,670,016 uops** | **Port 5** | **0.153 ms (+0.95 bips)** |

---

## 3. Decision: Stop Condition (Near-Miss)

- **Mandatory Promotion Floor:** $\ge +100\text{ bips} \approx 1.6\text{ ms}$ on the 160.5 ms crown baseline.
- **Admission Threshold:** $\ge 1.5\text{ ms}$ source-counted deletion ceiling.
- **Computed Upper Bound:** **0.153 ms (3.67 M uops, +0.95 bips)**.
- **Outcome:** The ranked witness transposition path `tr8x16_zmm` is already at the optimal 24-permute theoretical lower bound for 2-input vector permutes. The ancillary `tr8` path offers at most 0.153 ms of deletion ceiling, which is $< 1.5\text{ ms}$.
- **Status:** **NEAR-MISS REFUTATION — STOPPED EARLY PER OUTPUT CONTRACT.**
