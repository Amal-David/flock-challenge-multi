# Proposal: Additive NTT Fused-Four Butterfly Twiddle-Broadcast Audit & Deletion Ceiling

## Executive Summary
- **Target Subsystem**: Additive NTT fused-four butterfly twiddle-broadcast setup (`butterfly_fused_4layer_row_impl`).
- **Mechanism Analyzed**: Hoisting or precomputing the 15-twiddle broadcast and $x^{64}$-companion derivation (`vbroadcasti32x4` / `vpslldq` / `vpclmulqdq`) per fused-four block.
- **Commit SHA**: `911528bacb0804913dc5a3e0438b1d41ce0f10ec`
- **Counted Invocations**: 262,144 invocations per proof.
- **Port-5 Uop Deletion**: 45 Port-5 uops per invocation = 11,796,480 uops per proof.
- **Estimated Timing Delta**: **0.4915 ms** (~30.6 bips) on Port 5, with a whole-instruction ceiling of **0.9830 ms** (~61.2 bips).
- **Decision**: **NEAR-MISS / STOP**. The ceiling (0.49 ms) is strictly below the mandatory promotion threshold ($\ge 1.50\text{ ms} = +100\text{ bips}$). Per contract, stops early without speculative implementation.

## 1. Mechanism & Target Citations
- **Source Target**:
  - `crates/flock-core/src/ntt/additive_ntt_f128/kernels/x86_64.rs:1678-1682` (`butterfly_fused_4layer_row_impl`):
    Broadcasts and computes split companion for 15 twiddles via `tw_x4::<false, DIET>(*value)`.
  - `crates/flock-core/src/ntt/additive_ntt_f128/kernels/x86_64.rs:38-50` (`tw_x4`):
    Emits `_mm512_broadcast_i32x4` + `ghash_shift64_x4` (`_mm512_bslli_epi128::<8>` + `_mm512_clmulepi64_epi128::<0x01>` + `_mm512_xor_si512`).
  - `crates/flock-core/src/ntt/additive_ntt_f128.rs:1499,2266,2450,2921,3481,3520`:
    Callers in top and deep NTT passes.
- **Assembly Location**:
  - `target/x86_64-unknown-linux-gnu/challenge/deps/flock_core-*.s`:
    Symbol `_RNvNtNtNtCsfhNHrMHPExr_10flock_core3ntt17additive_ntt_f1287kernels26butterfly_fused_4layer_row` (lines 839229–839320, block `.LBB2046_27`).

## 2. Counted Deletion & Ceiling Estimation
- **Assembly Instructions in Setup Block (`.LBB2046_27`)**:
  - 15 × `vbroadcasti32x4` (Port 5)
  - 15 × `vpslldq $8` (Port 5)
  - 15 × `vpclmulqdq $1` (Port 5)
  - 15 × `vpxorq` (Port 0/1/5)
  - 30 × `vmovups`/`vmovdqu64` (stack stores to 3176-byte frame)
  - 2 × `movq`
- **Inner Loop (`.LBB2046_33`)**:
  - 476 instructions per 16-lane iteration step (32 butterflies).
  - Contains zero removable shuffles/permutes (`vpermq`, `vpshufd`, `vpunpck`, `vpblend`).
- **Ceiling Calculations**:
  - Port-5 uops / invocation: $15 + 15 + 15 = 45\text{ uops}$.
  - Total Port-5 uops / proof: $45 \times 262,144 = 11,796,480\text{ uops} \approx 11.80\text{ M uops}$.
  - At $24\text{ M uops/ms}$ on saturated Port 5:
    $$\text{Ceiling} = \frac{11,796,480}{24,000,000} = \mathbf{0.4915\text{ ms}} \approx 30.6\text{ bips}$$
  - Upper bound including all 90 setup uops:
    $$\text{Max} = \frac{90 \times 262,144}{24,000,000} = \mathbf{0.9830\text{ ms}} \approx 61.2\text{ bips}$$

## 3. Correctness & Rollback
- Bit-identical reference behavior preserved.
- Rollback flag definition: `FLOCK_NO_NTT_TW_HOIST=1` (incumbent path passes `&[F128; 15]` and computes twiddles in kernel preamble).

## 4. Verification & Gates
- **Cross-target check (x86_64)**:
  `RUSTFLAGS="-C target-feature=+avx512f,+avx512bw,+avx512vl,+avx512dq,+avx512vbmi,+vpclmulqdq,+gfni,+bmi2,+avx2,+aes,+pclmulqdq,+sse4.2" cargo check --target x86_64-unknown-linux-gnu --profile challenge --workspace`
  **PASSED** (exit code 0).
- **Native tests**:
  `cargo test -p flock-core --lib`
  **PASSED** (all 58 NTT tests passed; 449 unit tests passed).
- **Assembly audit**:
  Emitted x86_64 assembly verified in `target/x86_64-unknown-linux-gnu/challenge/deps/flock_core-*.s`.
  - Inner loop instruction count: 476 instructions.
  - Stack frame: `subq $3176, %rsp` (3,176 bytes).

## 5. Residual Risks & Next Steps
- Sub-kernel micro-optimizations on NTT fused-four twiddle broadcast cannot bridge the $1.50\text{ ms}$ line on their own.
- Campaign recommendations: Pivot research to higher-residual phases (StreamProj witness generation at 14.6% or Lincheck / Zerocheck representation changes).
