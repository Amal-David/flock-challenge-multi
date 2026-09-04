# GHASH Split-Twiddle Migration Call Site Audit and Counts

## Summary

* **Target Architecture**: x86_64 AVX-512 + VPCLMULQDQ (Sapphire Rapids c7i)
* **Ranked Shape**: $\ell = 64$, $N_{\text{medium}} = 4$, $m = 25$, Round-1 AB processes $524,288$ windows ($2^{19}$).
* **Mechanism**: Replace `ghash_mul_x4(v, t)` (6 CLMUL) with `ghash_mul_x4_split(v, t, t_x64)` (5 CLMUL) where $t_{\text{x64}} = \text{ghash\_shift64\_x4}(t)$ ($t \cdot x^{64} \pmod p$) is computed once outside the loop for loop-invariant multipliers $t$.
* **Rollback Flag**: `FLOCK_NO_ZC_GHASH_SPLIT=1`

---

## Enumeration of Call Sites in Target Files

### 1. `crates/flock-core/src/zerocheck/univariate_skip_optimized/kernels/x86_64.rs`

| File : Line | Enclosing Function | Multiplier Operand | Loop Context | Eligible? | Reason | Calls / Proof (Ranked Shape) | CLMULs Deleted / Proof |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| `kernels/x86_64.rs:796` | `accumulate_convert_ab_x86_avx512` | `eq` | `for lane in (0..ELL).step_by(4)` (16 iterations) | **YES** | `eq` is broadcast from `eq_lo_val` outside loop; constant for all 16 iterations. | 8,388,608 (when active under `FLOCK_NO_AB_NIBBLE=1`) | 7,864,320 (when active) |
| `kernels/x86_64.rs:956` | `accumulate_convert_ab_x86_avx512_nibble` (`scaled0`) | `eq` | `for lane_base in (0..ELL).step_by(16)` (4) $\times$ `for group in 0..2` (2) | **YES** | `eq` is broadcast from `eq_lo_val` outside loop; constant for all 8 iterations. | 4,194,304 | 3,932,160 |
| `kernels/x86_64.rs:957` | `accumulate_convert_ab_x86_avx512_nibble` (`scaled1`) | `eq` | `for lane_base in (0..ELL).step_by(16)` (4) $\times$ `for group in 0..2` (2) | **YES** | `eq` is broadcast from `eq_lo_val` outside loop; constant for all 8 iterations. | 4,194,304 | 3,932,160 |

**Subtotal for Round 1 AB Production Default (`nibble` kernel)**:
- Windows per proof: $524,288$ ($2^{19}$).
- Total `ghash_mul_x4` calls converted: $524,288 \times 16 = 8,388,608$.
- Total `ghash_shift64_x4` shifts hoisted: $524,288 \times 1 = 524,288$ (1 per window).
- Net CLMUL instructions deleted: $8,388,608 - 524,288 = 7,864,320$ CLMUL.

---

### 2. `crates/flock-core/src/zerocheck/multilinear/kernels/x86_64.rs`

| File : Line | Enclosing Function | Multiplier Operand | Loop Context | Eligible? | Reason | Calls / Proof (Ranked Shape) | CLMULs Deleted / Proof |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| `kernels/x86_64.rs:108` | `fold_and_message_x86_avx512` -> `fold_x4` | `r` | Fallback arm of `if split` | — | Already migrated with `zc_fold_split_enabled()` rollback arm. | — | — |
| `kernels/x86_64.rs:162` | `fold_and_message_x86_avx512` (reorder) | `b1` | Stream message compute | **NO** | `b1` varies per iteration (loaded/folded intermediate state). Per-call shift would add 1 CLMUL and regress. | 0 | 0 |
| `kernels/x86_64.rs:163` | `fold_and_message_x86_avx512` (reorder) | `b0^b1` | Stream message compute | **NO** | Operands vary per iteration. | 0 | 0 |
| `kernels/x86_64.rs:210` | `fold_and_message_x86_avx512` (non-reorder) | `b1` | Stream message compute | **NO** | Operands vary per iteration. | 0 | 0 |
| `kernels/x86_64.rs:211` | `fold_and_message_x86_avx512` (non-reorder) | `b0^b1` | Stream message compute | **NO** | Operands vary per iteration. | 0 | 0 |
| `kernels/x86_64.rs:648-651` | `round2_lookahead_chunk_x86_avx512` -> `eq_weights!` | `w` | Fallback arm of `if wsplit` | — | Already migrated with `zc_wsplit_enabled()` rollback arm. | — | — |
| `kernels/x86_64.rs:901` | `round2_lookahead_chunk_x86_avx512` (`is_one`) | `w` | Fallback arm of `if wsplit` | — | Already migrated with `zc_wsplit_enabled()` rollback arm. | — | — |
| `kernels/x86_64.rs:958` | `round2_lookahead_chunk_x86_avx512` (`is_sparse`) | `w` | Fallback arm of `if wsplit` | — | Already migrated with `zc_wsplit_enabled()` rollback arm. | — | — |
| `kernels/x86_64.rs:1092` | `round2_lookahead_chunk_x86_avx512` (`BAKE` epilogue) | `lane_w` | `for i in 0..8` chunk reduction | **YES** | `lane_w` is loaded once per chunk from `bake.unwrap().lane`; constant across all 8 iterations. | 512 (64 chunks $\times$ 8) | 448 ($64 \times (8 - 1)$) |
| `kernels/x86_64.rs:1168` | `fold2_and_message_x86_avx512` -> `fold_regs` | `r1`/`r2` | Composed double fold loop | **YES** | `r1` and `r2` are broadcast constants outside the loop. | 0 (Lookahead cascade is default) | 0 (49,152 if active) |
| `kernels/x86_64.rs:1214` | `fold2_and_message_x86_avx512` | `b1` | Stream message compute | **NO** | Operands vary per iteration. | 0 | 0 |
| `kernels/x86_64.rs:1215` | `fold2_and_message_x86_avx512` | `b0^b1` | Stream message compute | **NO** | Operands vary per iteration. | 0 | 0 |
| `kernels/x86_64.rs:1312` | `fold2_and_message_lookahead_x86_avx512` -> `fold_regs` | `ra`/`rb` | Cascade lookahead fold loop | **YES** | `ra` and `rb` are broadcast constants outside the loop. | 65,535 (Rounds 5..25) | 65,407 ($65,535 - 128$) |
| `kernels/x86_64.rs:1493-1496` | `fold2_and_message_lookahead_x86_avx512` -> `transpose4` | `w` | Fallback arm of `if wsplit` | — | Already migrated with `zc_wsplit_enabled()` rollback arm. | — | — |
| `kernels/x86_64.rs:2108` | `fold2_from_packed_lookahead_x86_avx512` -> `fold_regs` | `r1`/`r2` | Packed lookahead fold loop | **YES** | `r1` and `r2` are broadcast constants outside the loop. | 0 (GFNI batch fold is default) | 0 (~65,536 if active) |
| `kernels/x86_64.rs:2207-2214` | `group_from_packed` (`ta_lo`, `ta_hi`, `tb_lo`, `tb_hi`) | `r1` | Level-1 pair fold | **YES** | `r1` is a broadcast constant outside the loop. | 0 (GFNI batch fold is default) | 0 (~131,072 if active) |
| `kernels/x86_64.rs:2587-2590` | `groups_general` | `w` | Fallback arm of `if wsplit` | — | Already migrated with `zc_wsplit_enabled()` rollback arm. | — | — |

---

## Grand Totals (Production Ranked Shape)

| Kernel / Site | Active Calls Converted / Proof | Shifts Hoisted / Proof | Net CLMULs Deleted / Proof | Saturated Port-5 Time Saved (@ 24 M uops/ms) |
| :--- | :---: | :---: | :---: | :---: |
| **Round-1 AB Nibble Convert** (`kernels/x86_64.rs:956,957`) | 8,388,608 | 524,288 | 7,864,320 | **0.3277 ms** (327.7 $\mu$s) |
| **Round-2 BAKE Epilogue** (`kernels/x86_64.rs:1092`) | 512 | 64 | 448 | **0.000019 ms** (18.7 ns) |
| **Rounds 5..25 Cascade Lookahead** (`kernels/x86_64.rs:1312`) | 65,535 | 128 | 65,407 | **0.0027 ms** (2.7 $\mu$s) |
| **Total Production Default** | **8,454,655** | **524,480** | **7,930,175** | **0.3304 ms** (330.4 $\mu$s) |

### Derivation Notes & Citations
1. **Round 1 AB**: At ranked shape $N_{\text{medium}} = 4$, $m = 25$, $k_{\text{skip}} = 6$, $N_{\text{large}} = 19$, the univariate skip table applies over $2^{19} = 524,288$ windows. For each window, `accumulate_convert_ab_x86_avx512_nibble` processes $\text{ELL} = 64$ output lanes in 4 blocks of 16, with 2 groups per block, computing 2 vector multiplies per group = 16 4-lane `ghash_mul_x4` calls per window. Hoisting `eq64 = ghash_shift64_x4(eq)` reduces 6 CLMULs $\to$ 5 CLMULs on all 16 calls at the cost of 1 shift outside the loop, saving $16 - 1 = 15$ CLMULs per window.
2. **Round 2 BAKE**: Processes 64 worker chunks ($hi\_size = 64$). At chunk completion, `acc[i].reduce_lanes()` is scaled by `lane_w` for $i \in 0..8$. Hoisting `lane_w64` saves $8 - 1 = 7$ CLMULs per chunk.
3. **Rounds 5..25 Cascade Lookahead**: Evaluates $(2^{17} + 2^{15} + \dots + 2) / 4 = 43,690$ groups of 4 outputs. Each group evaluates `fold16_to_4` twice (A and B sides), invoking `fold_regs` 3 times per side = 1.5 `fold_regs` calls per output element = 65,535 calls total across all cascade tail rounds.
4. **Throughput Scaling**: On Intel Sapphire Rapids, VPCLMULQDQ-zmm and VPSLLDQ-zmm both issue exclusively on port 5 (throughput cap $\approx 24\times 10^6$ uops/ms at nominal all-core turbo). Deleting 7.93 M port-5 CLMUL uops removes $\approx 0.330$ ms of saturated port-5 execution time.
