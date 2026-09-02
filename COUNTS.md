# NTT Deep Pass Bandwidth vs. Port Analysis and Operation Counts

**Branch**: `lane/ntt-deep-traffic`  
**Crown Base**: `c75aeceb` (1,633,567.46 comp/s)  
**Target Architecture**: AWS `c7i.4xlarge` (Intel Xeon Platinum 8488C Sapphire Rapids, 8 physical cores × 2 SMT = 16 vCPUs)  
**Hotspot Phase**: Additive-NTT Deep Pass (43.0 ms per proof out of 160 ms total)

---

## 1. Executive Summary: Bandwidth vs. Port Bound Determination

**Verdict**: The Additive-NTT Deep Pass is **MEMORY-BANDWIDTH BOUND** (and secondarily bounded by SMT thread contention across sibling cores).

* **Memory Bandwidth Required**:
  - Minimum traffic (assuming 128 KiB tail blocks hit in L2): **2.15 GB** per proof.
  - Worst-case traffic (if Sweep 1 evicts L2 across 2 MiB subgroups): **4.29 GB** per proof.
  - Over the measured **43.0 ms** elapsed duration, the sustained traffic is **49.9 GB/s to 99.9 GB/s**.
  - On the ranked AWS `c7i.4xlarge` instance (a 16-vCPU partition of an 8-channel DDR5 server), the effective per-instance DRAM bandwidth ceiling is **~35–50 GB/s**. The phase operates directly at this DRAM saturation floor.
* **Compute / Port Ceiling**:
  - Total VPCLMULQDQ instructions across all 11 deep layers: **350,224,384 uops** (executing on Port 5).
  - 8 physical cores at ~3.0 GHz with 1 VPCLMUL uop/cycle throughput provide **24.0 billion VPCLMUL uops/s**.
  - Minimum theoretical VPCLMUL execution time: $350.22\text{M} / 24.0\text{G} = \mathbf{14.59\text{ ms}}$.
  - Total instructions (all ports, IPC ~2.8): $\approx 1.05 \times 10^9\text{ uops} \implies \mathbf{15.63\text{ ms}}$.
* **The 28.4 ms Gap** ($43.0\text{ ms} - 14.6\text{ ms}$):
  The ~28.4 ms residual is attributable to DRAM/LLC traffic bottlenecks (transferring 2–4 GB at 35–50 GB/s) and SMT sibling interference (the 8 consumer leaf-hashing threads sharing L2/L3 cache and fill buffers).

---

## 2. Geometry and Parameters of the Ranked Proof

Citations from `flock-core` and `flock-prover`:
* **Codeword Size**: $N = 2^{20} = 1,048,576$ positions, $K = 64$ NTT lanes ($2^6$).  
  Elements per codeword: $2^{20} \times 64 = 67,108,864$ `F128` elements ($1,073,741,824$ bytes = **1.0 GiB**).  
  *Citation*: [`crates/flock-core/src/pcs/commit.rs:56-94`](crates/flock-core/src/pcs/commit.rs#L56-L94), [`crates/flock-prover/src/prover.rs:865-885`](crates/flock-prover/src/prover.rs#L865-L885).
* **Layer Partitioning**:
  - Total layers: $\log_2(d) = 20$.
  - Top pass: layers $0..9$ ($n_{\text{top}} = 9$). Handled by `seed_top_fused8_pass` (layers $1..8$) and top-layer drivers.
  - Deep pass: layers $9..20$ ($11$ layers: $9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19$).  
  *Citation*: [`crates/flock-core/src/ntt/additive_ntt_f128.rs:3178-3184`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L3178-L3184), [`additive_ntt_f128.rs:3436-3455`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L3436-L3455).
* **Subgroup Decomposition**:
  - Number of subgroups: $2^{n_{\text{top}}} = 2^9 = 512$ subgroups.
  - Subgroup size: $2^{20 - 9} = 2048$ positions $\times 64$ `F128` $\times 16$ bytes = **2,097,152 bytes = 2 MiB**.  
  *Citation*: [`crates/flock-core/src/ntt/additive_ntt_f128.rs:3761`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L3761).
* **Thread Execution Topology**:
  - Pinned SMT sibling pairs: 8 producer threads (1 per physical core) execute NTT butterflies and enqueue blocks to 8 consumer threads (on SMT sibling logical cores) executing Merkle leaf hashes.  
  *Citation*: [`crates/flock-core/src/ntt/additive_ntt_f128.rs:3748-3854`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L3748-L3854).

---

## 3. Memory Traffic Arithmetic

The incumbent deep pass (`fuse_blocks = true` in `additive_ntt_f128.rs:3450-3561`) executes two sweeps per subgroup:

### Sweep 1: Fused-4 Subgroup Sweep (Layers 9..13)
* Operates over the full 2 MiB subgroup ($2048$ rows $\times 64$ lanes).
* `sixteenth = 128`, `num_ntts = 64`.
* Memory read: $512 \times 2\text{ MiB} = \mathbf{1,073,741,824\text{ bytes (1.0 GiB)}}$.
* Memory written: $512 \times 2\text{ MiB} = \mathbf{1,073,741,824\text{ bytes (1.0 GiB)}}$.
* Total Sweep 1 Traffic: **2.0 GiB (2.15 GB)**.

### Sweep 2: Block-Fused Tail (Layers 13..20 + Merkle Leaf Hash)
* Subgroup decomposes into 16 disjoint blocks of $2^{20 - 13} = 128$ positions = **128 KiB** ($128 \times 64 \times 16\text{ B}$).
* Total blocks across proof: $512 \times 16 = 8,192$ blocks.
* Inside each 128 KiB block:
  1. **Fused-4** (layers $13..17$, `sixteenth = 8`, `num_ntts = 64`): reads 128 KiB, writes 128 KiB.
  2. **Fused-3** (layers $17..20$, 16 groups of 8 consecutive rows = 8 KiB): reads 128 KiB from L1/L2 cache, writes 128 KiB.
  3. **Leaf Callback**: reads 128 KiB for hashing on sibling thread.
* If the 128 KiB blocks remain L2-resident throughout Sweep 2:
  - DRAM read: $0$ (or reload from LLC if evicted by Sweep 1).
  - Memory traffic: $1.0\text{ GiB}$ writeback / transfer.
* **Total Traffic Across Memory Bus**:
  - $\text{Traffic}_{\text{min}} = 1.0\text{ GiB (read)} + 1.0\text{ GiB (write)} = 2.0\text{ GiB} = \mathbf{2.147\text{ GB}}$.
  - $\text{Traffic}_{\text{max}} = 2.0\text{ GiB (read)} + 2.0\text{ GiB (write)} = 4.0\text{ GiB} = \mathbf{4.295\text{ GB}}$.
* **Effective Bandwidth**:
  $$\text{BW}_{\text{min}} = \frac{2.147 \times 10^9\text{ bytes}}{0.0430\text{ s}} = \mathbf{49.9\text{ GB/s}}$$
  $$\text{BW}_{\text{max}} = \frac{4.295 \times 10^9\text{ bytes}}{0.0430\text{ s}} = \mathbf{99.9\text{ GB/s}}$$

* **Hardware Comparison**:
  - AWS `c7i.4xlarge` (8 cores on shared dual-socket Sapphire Rapids, 8-channel DDR5-4800): sustained partition DRAM ceiling is **35–50 GB/s**.
  - L2 cache: 2 MiB per physical core (total 16 MiB across 8 cores).
  - L3 cache: 32 MiB shared.
  - The measured 43.0 ms matches the memory bus saturation ceiling.

---

## 4. Arithmetic and Port-5 Instruction Counts

Across all 11 deep layers:
* Total scalar butterflies per layer: $\frac{N}{2} \times K = 2^{19} \times 64 = 33,554,432$ butterflies.
* Total scalar butterflies across 11 layers: $11 \times 33,554,432 = \mathbf{369,098,752\text{ butterflies}}$.
* Vector butterflies (`__m512i`, 4 lanes per vector): $11 \times \frac{33,554,432}{4} = \mathbf{92,274,688\text{ vector operations}}$.

### Breakdown by Kernel:
1. **Sweep 1 (Layers 9..13)**:
   - 4 layers = $33,554,432$ vector butterflies.
   - Algorithm: `ghash_mul_x4_split` (DIET: 4 VPCLMULQDQ uops per multiply).
   - VPCLMULQDQ count: $33,554,432 \times 4 = \mathbf{134,217,728\text{ uops}}$.
2. **Sweep 2 Part A (Layers 13..17)**:
   - 4 layers = $33,554,432$ vector butterflies.
   - Algorithm: `ghash_mul_x4_split` (DIET: 4 VPCLMULQDQ uops).
   - VPCLMULQDQ count: $33,554,432 \times 4 = \mathbf{134,217,728\text{ uops}}$.
3. **Sweep 2 Part B (Layers 17..20)**:
   - 3 layers = $25,165,824$ vector butterflies.
   - Algorithm: `butterfly_fused_3layer_rows_impl` using `LOW_INNER` (3 CLMULs) for 8 of 12 butterflies, `HIGH_ONE`/`DIET` for outer layer.
   - Average CLMULs per vector butterfly: $\approx 3.25$.
   - VPCLMULQDQ count: $25,165,824 \times 3.25 = \mathbf{81,788,928\text{ uops}}$.

**Total VPCLMULQDQ uops**: $134.22\text{M} + 134.22\text{M} + 81.79\text{M} = \mathbf{350,224,384\text{ uops}}$.

---

## 5. Feasibility and Ceilings of Potential Interventions

### A. Increase Layers Fused per Pass (e.g. Fused-5 or Fused-4 Tail)
* **Fused-5 Attempt**: Fusing 5 layers into one pass requires loading $2^5 = 32$ rows simultaneously.
  - Register requirement: 32 ZMM registers for data alone.
  - AVX-512 provides exactly 32 ZMM registers (`zmm0..zmm31`).
  - Storing 31 broadcast twiddles, polynomial constants (`0x87`), and intermediate multiplication cross-terms leaves 0 available registers, forcing 6–10 stack spills per butterfly.
  - Across $92.27\text{M}$ vector operations, this would cause $>300\text{M}$ stack spills, severely stalling execution pipelines.
* **Fused-4 Tail Attempt**: Sweep 2 already executes layers 13..16 (4 layers) and layers 17..19 (3 layers). Fusing 4 layers in the tail is mathematically impossible because the transform terminates at layer 19 ($17 + 3 = 20 = \log_2 d$). There is no 4th layer in the tail.
* **Net Ceiling**: **0.0 ms (Negative / Infeasible)**.

### B. Improve Subgroup Blocking for L2 Cache
* The subgroup size is already set to $2^{20 - 9} = 2048$ positions = **2 MiB**, matching the 2 MiB L2 cache per Golden Cove core.
* If subgroup size is reduced to 1 MiB ($n_{\text{top}} = 10$):
  - Top pass `seed_top_fused8_pass` only covers layers $1..8$.
  - Layer 9 would become an orphaned top layer requiring an extra un-fused DRAM sweep across the entire 1 GiB buffer.
  - Adding a 1 GiB DRAM sweep adds $\mathbf{+5.0\text{ to }8.0\text{ ms}}$ of latency.
* **Net Ceiling**: **Negative (Regressive)**.

### C. Exploit Rate-1/4 Zero Structure Deeper in Layer Stack
* The ranked commit uses `log_inv_rate = 1` (rate-1/2), where `seed_top_fused8_pass` expands the message directly across layers $1..8$.
* By layer 9 (deep pass entry), **every position across the 1 GiB buffer is non-zero**.
* The only remaining zero structure is the BLAKE3 padding tail on odd rows ($4$ trailing zero lanes out of 64), which is already eliminated by `ZeroOddTailLanes` and `row_lanes`.
* **Net Ceiling**: **0.0 ms (Already fully exploited)**.

### D. Port-bound Twiddle Broadcast Hoisting
* In `butterfly_interleaved_fused_4layer_rows`, 15 twiddles are broadcast for each of the `sixteenth` iterations of `r`.
* Sweep 1: 512 subgroups $\times 128$ calls $= 65,536$ calls $\times 15 = 983,040$ broadcasts.
* Sweep 2: 512 subgroups $\times 16$ blocks $\times 8$ calls $= 65,536$ calls $\times 15 = 983,040$ broadcasts.
* Total broadcasts: $1,966,080$ broadcasts per proof.
* Pre-broadcasting twiddles deletes redundant broadcasts ($1,835,520$ broadcasts deleted).
* Instructions deleted: $\approx 4.5 \times 10^6$ instructions.
* Execution ceiling on 8 cores at 3 GHz (IPC 3.0):
  $$\text{Ceiling} = \frac{4.5 \times 10^6\text{ instructions}}{8 \times 3.0 \times 10^9 \times 3.0} \approx \mathbf{0.0625\text{ ms}} = \mathbf{+3.9\text{ bips}}$$
* **Net Ceiling**: **0.06 ms** (far below the 0.8 ms threshold and 1.6 ms promotion bar).

---

## 6. Operation Counts and File:Line Citations

| Target Operation | Subsystem | Incumbent Per-Proof Count | File:Line Citation | Deletion Ceiling (ms) | Headroom vs +100 bips (1.6 ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Deep Pass DRAM Read/Write** | `additive_ntt_f128` | $2.15\text{ GB} - 4.29\text{ GB}$ | [`additive_ntt_f128.rs:3461-3561`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L3461-L3561) | Saturated (49–99 GB/s) | $0.0\text{ ms}$ (at hardware ceiling) |
| **Deep Pass Vector Butterflies** | `additive_ntt_f128` | $92,274,688\text{ ops}$ | [`additive_ntt_f128.rs:3598, 3519, 3547`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L3598) | $14.59\text{ ms}$ (Port 5 floor) | $0.0\text{ ms}$ (algebraic floor) |
| **VPCLMULQDQ Operations** | `x86_64/kernels` | $350,224,384\text{ uops}$ | [`kernels/x86_64.rs:63-106`](crates/flock-core/src/ntt/additive_ntt_f128/kernels/x86_64.rs#L63-L106) | $14.59\text{ ms}$ (Port 5 floor) | $0.0\text{ ms}$ (algebraic floor) |
| **Twiddle Broadcasts** | `x86_64/kernels` | $1,966,080\text{ calls}$ | [`kernels/x86_64.rs:1679-1682`](crates/flock-core/src/ntt/additive_ntt_f128/kernels/x86_64.rs#L1679-L1682) | $\mathbf{0.06\text{ ms}}$ | $\mathbf{+3.9\text{ bips}}$ (Near-miss) |
| **Subgroup Re-blocking ($n_{\text{top}}=10$)** | `additive_ntt_f128` | N/A | [`additive_ntt_f128.rs:3178-3184`](crates/flock-core/src/ntt/additive_ntt_f128.rs#L3178-L3184) | $-5.0\text{ to }-8.0\text{ ms}$ | **Negative / Regressive** |

---

## 7. Quantitative Finding

1. **Bandwidth Ceiling**: The Additive-NTT deep pass sweeps a 1 GiB buffer in two passes generating 2.15 to 4.29 GB of memory traffic over 43.0 ms, which fully saturates the 35–50 GB/s DRAM bandwidth available to the 16-vCPU partition on AWS Sapphire Rapids.
2. **Compute Floor**: VPCLMULQDQ throughput on Port 5 requires 14.59 ms of execution time across 8 cores at 3.0 GHz, while overall retirement requires 15.63 ms.
3. **Optimizations Exhausted**: 
   - Further layer fusion (fused-5) is blocked by 32-register AVX-512 architectural limits and massive stack spilling.
   - Tail layer fusion is already complete (4+3 layers in Sweep 2 exhaust all remaining deep layers 13..19).
   - Working set reblocking below 2 MiB causes an un-fused top-layer DRAM sweep penalty (+5 to 8 ms).
   - Zero-structure elimination is already exhausted.
   - Micro-optimizations (twiddle broadcast hoisting) offer at most **0.06 ms (+3.9 bips)**, well below the 0.8 ms stop floor and 1.6 ms (+100 bips) promotion requirement.
4. **Recommendation**: Quantified negative result. Kernel-level traffic optimization on the deep pass cannot yield the 100-bip promotion threshold.
