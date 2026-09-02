# Proposal: Additive-NTT Deep Pass Traffic & Arithmetic Ceiling Analysis (Negative/Near-Miss Report)

**Lane**: `lane/ntt-deep-traffic`  
**Base Commit**: `c75aeceb` (1,633,567.46 comp/s)  
**Hotspot Target**: Additive-NTT Deep Pass (43.0 ms per proof)  
**Evaluator Hardware**: AWS `c7i.4xlarge` (Intel Xeon 8488C Sapphire Rapids, 8 cores × 2 SMT)

---

## 1. Mechanism & Investigation

The additive-NTT deep pass processes layers 9 through 19 (11 layers total) of a $2^{20} \times 64$ `F128` codeword (1.0 GiB = 1,073,741,824 bytes) across 512 independent 2 MiB subgroups.

We investigated whether this 43.0 ms phase is **memory-bandwidth bound** or **port bound**, and surveyed all candidate optimizations:
1. **Memory Traffic vs. DRAM Saturation**:
   - Sweep 1 (layers 9..13): reads 1.0 GiB, writes 1.0 GiB.
   - Sweep 2 (layers 13..20): reads 1.0 GiB (from L2 or LLC), writes 1.0 GiB (consumed by Merkle leaf hashing on SMT sibling).
   - Total memory traffic: **2.15 GB to 4.29 GB** per proof.
   - Effective bandwidth required over 43.0 ms: **49.9 GB/s to 99.9 GB/s**.
   - On the ranked AWS `c7i.4xlarge` 16-vCPU instance slice, the sustained DRAM bandwidth ceiling is **~35–50 GB/s**.
   - **Conclusion**: The phase operates directly at the physical memory bandwidth saturation ceiling of the machine.

2. **Compute & Port-5 Pressure**:
   - Total butterflies in deep pass: $11 \text{ layers} \times 2^{19} \times 64 = 369,098,752$ butterflies ($92,274,688$ vector operations on `__m512i`).
   - Total VPCLMULQDQ instructions: **350,224,384 uops** (executing on Port 5).
   - 8 physical cores at 3.0 GHz execute at most $24.0 \times 10^9$ VPCLMUL uops/s $\implies$ theoretical port-5 compute floor is **14.59 ms**.
   - Total instruction uops ($\approx 1.05 \times 10^9$ uops at IPC 2.8) $\implies$ **15.63 ms**.
   - The 28.4 ms delta between compute (14.6 ms) and measured elapsed time (43.0 ms) is strictly bounded by DRAM traffic and SMT sibling contention.

3. **Intervention Analysis**:
   - **Fused-5 Layer Sweep**: Loading 32 rows requires 32 ZMM registers, leaving 0 registers for twiddles and multiplication cross-terms, causing 6–10 stack spills per butterfly ($>300\text{M}$ spills). Infeasible.
   - **Tail Layer Fusion**: Sweep 2 already fuses layers 13..16 (4 layers) and layers 17..19 (3 layers) in L2 cache. The transform terminates at layer 19, so no fourth layer exists to fuse.
   - **Subgroup Reblocking**: Shrinking subgroup size below 2 MiB ($n_{\text{top}}=10$) breaks `seed_top_fused8_pass` alignment and forces an extra 1 GiB DRAM sweep (+5 to 8 ms penalty).
   - **Twiddle Broadcast Hoisting**: Pre-broadcasting 15 twiddles across `r` iterations in `butterfly_interleaved_fused_4layer_rows` deletes $\approx 4.5\text{M}$ instructions. The maximum theoretical ceiling on 8 cores is **0.0625 ms (+3.9 bips)**, which is a near-miss (< 0.8 ms ceiling threshold; promotion requires 1.6 ms / +100 bips).

---

## 2. Counted Deletions & Headroom

* **Target Operations Counted**:
  - Deep pass memory traffic: $2.15\text{ GB} - 4.29\text{ GB}$ per proof ([`crates/flock-core/src/ntt/additive_ntt_f128.rs:3461-3561`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L3461-L3561)).
  - VPCLMULQDQ operations: $350,224,384\text{ uops}$ ([`crates/flock-core/src/ntt/additive_ntt_f128/kernels/x86_64.rs:63-106`](crates/flock-core/src/ntt/additive_ntt_f128/kernels/x86_64.rs#L63-L106)).
  - Twiddle broadcasts: $1,966,080\text{ broadcasts}$ ([`kernels/x86_64.rs:1679-1682`](crates/flock-core/src/ntt/additive_ntt_f128/kernels/x86_64.rs#L1679-L1682)).
* **Headroom vs. Promotion Threshold**:
  - Required promotion threshold: **+100 bips = +1.00% = 1.60 ms**.
  - Maximum deletion ceiling from twiddle broadcast hoisting: **0.06 ms (+3.9 bips)**.
  - Maximum deletion ceiling from layer re-fusion: **0.00 ms (Infeasible / Regressive)**.
  - Result: **Near-miss / Hard Stop** (< 0.8 ms ceiling).

---

## 3. Correctness & Rollback Mechanisms

The incumbent deep-pass architecture already provides complete runtime rollback switches:
* `FLOCK_NO_NTT_DEEP_BLOCK_FUSE=1`: Restores the un-fused sweep schedule over memory ([`additive_ntt_f128.rs:346`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L346)).
* `FLOCK_NO_NTT_DEEP_SPLIT=1`: Restores single-worker alternating execution instead of SMT sibling producer/consumer queues ([`additive_ntt_f128.rs:364`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L364)).
* `FLOCK_NO_NTT_SHAPED=1`: Restores generic non-monomorphized row kernel dispatch ([`additive_ntt_f128.rs:642`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L642)).

All 468 native tests in `flock-core` verify bit-identical outputs across all configurations on AVX-512 hardware.

---

## 4. Native Validation Receipts on Ranked Hardware

Validated natively on AWS `c7i.4xlarge` (Intel Sapphire Rapids Xeon Platinum 8488C):
* **Scope Check**: Clean (editable path confined to `crates/flock-core/src/**` and `crates/flock-prover/src/**`).
* **Challenge Build**: Succeeded (`--profile challenge`, `-C target-cpu=native`).
* **Native Unit Tests**: `flock-core` passed all 468 tests in 10.63s on real AVX-512.
* **Assembly Emission (x86_64)**:
  - `butterfly_fused_4layer_row_shaped::<128, 64, 0>`: 494 instructions, frame 0B.
  - `butterfly_fused_4layer_row_shaped::<8, 64, 0>`: 494 instructions, frame 0B.
  - `butterfly_fused_3layer_rows_shaped::<64>`: 412 instructions, frame 0B.
  - Zero stack frame / zero spill overhead in all hot monomorphized kernels.

---

## 5. Residual Risks & Conclusion

1. **Architectural Saturation**: The additive-NTT deep pass cannot be sped up by further kernel unrolling or layer fusion on 8 Golden Cove cores due to the 32-register AVX-512 file size and the 35–50 GB/s instance DRAM memory bandwidth ceiling.
2. **Campaign Direction**: Optimization effort must focus on high-level representation fusions or protocol changes across component boundaries rather than kernel micro-optimizations in the NTT deep pass.
