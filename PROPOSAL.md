# Proposal: Zerocheck Round-One AB Plane Accumulation & Register Census

## 1. Mechanism Summary

This investigation evaluates the round-one AB plane store+load round-trips in zerocheck (`crates/flock-core/src/zerocheck/univariate_skip_optimized.rs` and `kernels/x86_64.rs`). In the crown (`c75aeceb`), round-one AB executes $524,288$ window evaluations across 128 $x_{hi}$ bands ($4,096$ windows per band). In each window, the kernel `convert_ab_nomul_x86_gfni_direct` loads $14$ (or $15$) input rows directly from memory into $14$ ZMM registers (`zmm0..zmm13`), loops over $16$ byte planes, evaluates GFNI bit-matrix transforms, and updates the plane accumulation in memory.

We evaluated:
1. Eliminating the intermediate plane load/store round-trips into registers.
2. Holding $16$ plane accumulators live in ZMMs (row-major streaming).
3. Splitting into $8$-plane passes ($2$ passes over rows).

---

## 2. Counted Deletion with Source Citations

- **Source Files & Citations**:
  - `crates/flock-core/src/zerocheck/univariate_skip_optimized.rs:3339-3611` (`process_one_x_hi_ab_only`)
  - `crates/flock-core/src/zerocheck/univariate_skip_optimized.rs:3742-3752` (`eq_fold_state` parameters: $n_{lo} = 12$, $bank\_bits = 7$, $hi\_size = 128$, $big\_lo\_size = 4096$)
  - `crates/flock-core/src/zerocheck/univariate_skip_optimized/kernels.rs:610-645` (`convert_ab_nomul_gfni_direct`)
  - `crates/flock-core/src/zerocheck/univariate_skip_optimized/kernels/x86_64.rs:1930-2007` (`convert_ab_nomul_x86_gfni_direct`)

- **Per-Proof Invocations**:
  - Total window kernel calls: $128 \times 4096 = 524,288$ windows
  - Parity 0: $(first, n) = (2, 16)$ $\rightarrow 14$ rows ($262,144$ invocations)
  - Parity 1: $(first, n) = (0, 15)$ $\rightarrow 15$ rows ($262,144$ invocations)

- **Uops Targeted for Deletion**:
  - Plane accumulator loads: $8,126,464$ vector loads ($31/32 \times 524,288 \times 16$)
  - Plane accumulator stores: $8,388,608$ vector stores ($524,288 \times 16$)
  - **Total Deletable Memory Uops**: $\mathbf{16,515,072\text{ uops}}$ ($\approx 16.52\text{ M uops}$)
  - Mandatory GFNI transforms: $121,634,816$ instructions (`vgf2p8affineqb`, cannot be pruned)
  - Accumulator boolean operations: $62,914,560$ uops (`vpternlogq` / `vpxord`)

---

## 3. Time Ceiling Estimation & Stop Condition

- **Hardware**: AWS `c7i.4xlarge` (Intel Xeon 8488C Sapphire Rapids, 8 cores $\times$ 2 SMT @ $\approx 3.0\text{ GHz}$)
- **Saturated Port Throughput**: $\approx 24\text{ M uops/ms}$
- **Upper-Bound Time Deletion**:
  $$\text{Ceiling} = \frac{16,515,072\text{ uops}}{24,000,000\text{ uops/ms}} = \mathbf{0.688\text{ ms}} \approx \mathbf{0.69 - 0.70\text{ ms}}$$
- **Baseline Proof Time (Crown `c75aeceb`)**: $160.5\text{ ms}$
- **Promotion Threshold (+100 bips)**: Requires $\ge 1.605\text{ ms}$ reduction.
- **Maximum Reachable Gain**: $+42.9\text{ bips} \ll +100\text{ bips}$.
- **Stop Condition Trigger**: Because the ceiling is **$0.69\text{ ms} < 1.50\text{ ms}$**, Output Contract Item 1 mandates recording the near-miss refutation and stopping early.

---

## 4. Correctness Argument & Unit Tests

The mathematical transformation computes:
$$\text{bank}[u][k] = \sum_{w=0}^{31} \text{GFNI}(\text{rows}(w \cdot 128 + u), \text{mats}[w][\cdot, k])$$
GFNI affine byte multiplication over $\text{GF}(2^8)$ is distributive and linear over XOR addition. Reassociating or caching row loads preserves bit-for-bit exactness.

- Unit test added: `test_r1_direct_matches_staged_shapes` in `crates/flock-core/src/zerocheck/univariate_skip_optimized/kernels/x86_64.rs`.
- Validates direct GFNI output against staged reference across all combinations:
  - $(first, n) = (2, 16)$ (14 live rows)
  - $(first, n) = (0, 15)$ (15 live rows)
  - $\text{FIRST\_WRITE} \in \{\text{true}, \text{false}\}$
  - Poisoned destination memory buffers.
- Asserts byte equality on all outputs.

---

## 5. Rollback Switch

- **Environment Variable**: `FLOCK_NO_ZC_R1_AB_DIRECT=1`
- **Location**: `crates/flock-core/src/zerocheck/univariate_skip_optimized.rs:3811`
- **Behavior**: When set, bypasses the direct GFNI leaf and restores the incumbent staged path (`accumulate_convert_ab_nomul_gfni_range2` / `write_convert_ab_nomul_gfni_range2`) with complete exactness.
- **Commit SHA**: `25d5ee8a8cb9ba25cba8971f1146747b0a3952f4`

---

## 6. Gates & Assembly Verification

### A. Cargo Cross-Check Gate
```bash
export RUSTC=$HOME/.rustup/toolchains/1.97.0-aarch64-apple-darwin/bin/rustc CC_x86_64_unknown_linux_gnu=clang
RUSTFLAGS="-C target-feature=+avx512f,+avx512bw,+avx512vl,+avx512dq,+avx512vbmi,+vpclmulqdq,+gfni,+bmi2,+avx2,+aes,+pclmulqdq,+sse4.2" cargo check --target x86_64-unknown-linux-gnu --profile challenge --workspace
```
- **Result**: PASSED (exit code 0, workspace clean).

### B. Native Tests Gate
```bash
cargo test -p flock-core --lib zerocheck -- --test-threads=1
```
- **Result**: PASSED (105 tests passed, 0 failed).

### C. Assembly Check
Emitted x86-64 assembly for `convert_ab_nomul_x86_gfni_direct<2, 16, false>`:
- **Stack Frame Size (`sub rsp, N`)**: **`0 bytes`** (no `subq %rsp` instruction emitted; no register spills to stack).
- **Inner Loop Instructions**: 25 instructions per plane iteration ($14\times \text{vgf2p8affineqb}$, $7\times \text{vpternlogq}$, $1\times \text{vmovdqu64}$, $1\times \text{addq}$, $1\times \text{cmpq}$, $1\times \text{jne}$). Total unrolled loop body = 400 instructions.
- **Stack Frame Growth Gate**: $0\text{ bytes} \le 256\text{ bytes}$ (PASSED).

---

## 7. Residual Risks & Refutation Rationale

1. **Hard Deletion Ceiling**: The entire memory traffic of intermediate plane stores and loads is $16.52\text{ M uops} = 0.69\text{ ms}$, capping maximum throughput gain at $+43\text{ bips}$.
2. **Register Exhaustion in Row-Major Layout**: Holding $16$ plane accumulators in ZMMs requires $16$ registers. Streaming rows through $16$ matrices serializes accumulator updates. When tested at commit `18b23971`, this regressed by $-5.27\text{ bips}$ (`lessons.md:32`).
3. **8-Plane Split Drawback**: Splitting into $8$-plane passes requires loading all $14$ rows twice ($28$ vector loads vs $14$), and doubles boolean XOR count ($224$ vs $112$), yielding strictly negative net instruction efficiency.
4. **Conclusion**: Zerocheck Round-1 AB is already operating at the hardware/algebraic floor on `c75aeceb`. This candidate is refutable and closed as a near-miss.
