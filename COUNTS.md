# NTT Fused-Four Butterfly Twiddle-Broadcast Audit & Counts

## Target & Invocations
- **Source Target**: Additive NTT fused-four butterfly kernel twiddle setup, `crates/flock-core/src/ntt/additive_ntt_f128/kernels/x86_64.rs:1678-1682` (`butterfly_fused_4layer_row_impl`), called through `butterfly_fused_4layer_row`, `butterfly_fused_4layer_row_pf`, and `butterfly_fused_4layer_row_shaped` from `crates/flock-core/src/ntt/additive_ntt_f128.rs:1499,2266,2450,2921,3481,3520`.
- **Ranked Invocations Per Proof**: 262,144 kernel invocations.

## Emitted Assembly Analysis
- **Assembly Artifact**: `target/x86_64-unknown-linux-gnu/challenge/deps/flock_core-*.s`
- **Symbol**: `_RNvNtNtNtCsfhNHrMHPExr_10flock_core3ntt17additive_ntt_f1287kernels26butterfly_fused_4layer_row` (and monomorphizations `butterfly_fused_4layer_row_shaped<128, 64, H>`)
- **Stack Frame**: `subq $3176, %rsp` (3,176 bytes)
- **Inner Loop**: `.LBB2046_33` (476 instructions per 16-lane iteration step, unrolled across 32 butterflies)
  - Contains **zero** removable `vpermq`, `vpshufd`, `vpunpck`, or `vpblend` instructions.
  - Contains **one** loop-invariant `vpbroadcastq` for the reduction constant `0x87`.
- **Setup Block**: `.LBB2046_27` (lines 839229–839320 in emitted asm):
  Executed on **every** invocation of `butterfly_fused_4layer_row_impl`:
  - 15 × `vbroadcasti32x4` (shuffle / broadcast uop on Port 5)
  - 15 × `vpslldq $8` (byte shift uop on Port 5)
  - 15 × `vpclmulqdq $1` (carry-less multiplication uop on Port 5)
  - 15 × `vpxorq` (XOR uop on Port 0/1/5)
  - 30 × `vmovups`/`vmovdqu64` (stack store uops to `%rsp` slots)
  - 2 × address arithmetic instructions (`movq`)

## Port-5 Uop Count & Deletion Ceiling
- **Port 5 Uops per invocation in setup**:
  - Broadcast (`vbroadcasti32x4`): 15 uops
  - Byte shift (`vpslldq`): 15 uops
  - CLMUL (`vpclmulqdq`): 15 uops
  - **Total Port 5 uops per invocation**: **45 uops**
- **Total Port 5 Uops per proof**:
  - $45 \text{ uops} \times 262,144 \text{ invocations} = 11,796,480 \text{ uops} \approx 11.80 \text{ M uops}$
- **Estimated Port-5 Timing Deletion**:
  - At $\sim 24 \text{ M uops/ms}$ on saturated Port 5 on Intel Xeon 8488C Sapphire Rapids:
  $$\text{Ceiling} = \frac{11,796,480 \text{ uops}}{24,000,000 \text{ uops/ms}} = \mathbf{0.4915 \text{ ms}} \approx 0.49 \text{ ms}$$
  - In score bips on 160.5 ms proof: $\frac{0.4915}{160.5} \times 10,000 \approx \mathbf{30.6 \text{ bips}}$

- **Total Uops Ceiling (all ports)**:
  - If all 90 setup instructions (including XORs, loads, and stack stores) were completely eliminated:
  $$90 \text{ uops} \times 262,144 = 23,592,960 \text{ uops}$$
  $$\text{Upper bound} = \frac{23,592,960}{24,000,000} = \mathbf{0.9830 \text{ ms}} \approx 0.98 \text{ ms} \approx 61.2 \text{ bips}$$

## Conclusion & Stop Decision
- The mandatory threshold for candidate admission is $\ge 1.50 \text{ ms}$ of timed proof reduction ($+100 \text{ bips}$).
- The strict Port-5 deletion ceiling for hoisting the entire 15-twiddle broadcast setup is **$0.4915 \text{ ms}$** ($\le 30.6 \text{ bips}$), with an all-port theoretical maximum of **$0.9830 \text{ ms}$**.
- Per the Output Contract ("if the ceiling is < 1.5 ms, write the near-miss and STOP"), this target is documented as a verified near-miss and stopped early.
