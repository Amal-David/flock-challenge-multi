//! **Prototype** — GFNI bit-matrix form of the §2.1 inverse-NTT/LDE row
//! extension (`InvNttTableByteSingleGf8::apply`), batched 64 rows per call.
//!
//! # The algebra
//!
//! `apply` is
//!
//! ```text
//!   out[i] = ⊕_{b=0..8} T₀[byte_b][i ⊕ 8b]
//! ```
//!
//! and `T₀` is XOR-composed over the set bits of its index from the eight
//! unit-column images `cols[t] = T₀[1 << t]` (`inv_table.rs` §"T_0[0] already
//! zero"), so
//!
//! ```text
//!   out[i] = ⊕_b ⊕_{t ∈ byte_b} cols[t][i ⊕ 8b]
//!          = ⊕_b  N[i ⊕ 8b] · byte_b
//! ```
//!
//! where `N[i']` is the 8×8 GF(2) matrix whose column `t` is the byte
//! `cols[t][i']`. Every output bit is a fixed XOR of input bits: the whole
//! extension is one F₂-linear map, and the `i ⊕ 8b` coordinate permutation
//! folds into the *matrix index*, costing nothing at run time. Only **64**
//! distinct matrices exist (512 bytes total) — the 512 (chunk, output-byte)
//! pairs are re-indexings of them.
//!
//! # The batching that makes GFNI apply
//!
//! `VGF2P8AFFINEQB` supplies one 8×8 matrix per QWORD, i.e. one matrix shared
//! by eight byte lanes. Our map needs a *different* matrix per output lane, so
//! broadcasting one input byte across a ZMM and varying the matrix per qword
//! yields only eight distinct outputs — that direction is a dead end.
//!
//! Batch across **rows** instead, exactly as
//! [`crate::zerocheck::multilinear::kernels::x86_64`]'s `gfni_fold64_regs`
//! does: `p[b]` is a *plane* — input byte `b` of 64 different rows — and the
//! matrix is `_mm512_set1_epi64(...)`, the same map applied to 64 independent
//! rows. One `vgf2p8affineqb` then produces one output byte of all 64 rows.
//! Per 64 rows the battery is 8 chunk planes × 64 output-byte planes = 512
//! affine products, i.e. **8 affine ops per row** for all 64 output bytes.
//!
//! # Layout
//!
//! Input: 64 rows × 8 packed bytes = 512 contiguous bytes (exactly eight
//! `b_med` blocks of the round-1 window, whose K-rows are contiguous).
//! Row `r` occupies `src[8r .. 8r+8]`.
//!
//! [`extend64_planes`] emits the 64 output-byte planes (`plane[i].byte[r] =
//! apply(row r)[i]`) — the form a fused a·b pipeline consumes directly,
//! because `VGF2P8MULB` and the `x^k` shift-reduce are both pointwise in
//! `(row, lane)` and so are layout-agnostic.
//!
//! [`extend64_rows`] additionally transposes those 64 planes back to row-major
//! (`out[64r + i]`), which is what a drop-in replacement for the incumbent
//! leaf would need. That 64×64 byte transpose is the single most uncertain
//! cost in the design, so it is measured separately on purpose.
//!
//! # Measured (real Sapphire Rapids assembly)
//!
//! Three `#[unsafe(no_mangle)]` probes in this file — the two kernels and the
//! incumbent two-image table apply over the *same* 64 rows — emitted under
//! `-C target-cpu=sapphirerapids` with the full AVX-512/GFNI feature set, same
//! `&`/`&mut` argument shapes so `noalias` is symmetric. Dynamic instruction
//! counts per 64 rows (= 64 applies), loop bodies multiplied by trip count:
//!
//! ```text
//!                            incumbent   plane-major   row-major
//!   instructions                  2174           961        1831
//!   per apply                    33.97         15.02       28.61
//!   ratio                            —         2.26x       1.19x
//!   port-5-only shuffles           208            32         480
//!   vgf2p8affineqb                   0           512         512
//!   port-0/5 vector ALU            400           256         256
//!   512-bit table/matrix reads     512    64 (8-byte)  64 (8-byte)
//!   GPR + control uops            1230            22          60
//! ```
//!
//! And for the fused pipeline ([`fused_ab8`]) against the incumbent leaf on
//! its current `_pidx` / `offw` path, per 64-byte medium window:
//!
//! ```text
//!                              incumbent   fused plane-major
//!   instructions                     551                 289
//!   port-5-only shuffles              54                  29
//!   GPR + control uops               314                 4.1
//!   512-bit table reads    128 (8 KiB)      8 (64 bytes)
//!   working set                  32 KiB           512 bytes
//! ```
//!
//! **Port 5 goes down, not up.** The plane-major form adds a fold stage and
//! the plane transposes (~29 shuffles/window), but it deletes the incumbent's
//! six per-Horner-iteration lane permutes (54/window). So this is a strict
//! deletion on every axis measured, not a shuffle-for-loads trade.
//!
//! Two results worth keeping:
//!
//! * **The output transpose costs more than the extension it serves.**
//!   `1831 - 961 = 870` instructions per 64 rows — 448 port-5 shuffles plus
//!   128 spill store/reloads and 128 register copies feeding the destructive
//!   `vpermt2q` — and it takes port 5 from 0.5 to 7.5 uops per apply. A
//!   row-major drop-in for the incumbent leaf is therefore only 1.19x and is
//!   not worth shipping; the value is in the plane-major form, which a fused
//!   a·b pipeline can consume without ever transposing.
//!
//! * **The incumbent's cost is not where its instruction count suggests.**
//!   57% of its stream (19.2 of 33.97 uops per apply) is scalar index
//!   arithmetic and loop control, and it reads 8 x 64 bytes of table per
//!   apply — 8 KiB per 64-byte output block, out of a 32 KiB two-image table
//!   shared by two SMT siblings in a 48 KiB L1. Its 34 uops/apply imply a
//!   ~5.7-cycle front-end bound and a 4-cycle load-port bound, against a
//!   profiled ~18.75 cycles/apply. This kernel's entire working set is the
//!   512-byte matrix block and it reads 8 bytes per apply.
//!
//! # Portability of the correctness argument
//!
//! The kernel body is written **once**, generic over a 512-bit vector trait
//! ([`V512`]) with two implementations: `__m512i` on x86-64 and a portable
//! `Bytes64` byte-array model of the same eight instructions
//! (`vpermb`, `vpunpcklqdq`, `vpunpckhqdq`, `vpermt2q`, `vpxorq`,
//! `vpternlogq`, `vgf2p8affineqb`, load/store). The tests drive the portable
//! instantiation against [`InvNttTableByteSingleGf8::apply`] as oracle, so the
//! exact shuffle network and matrix encoding are pinned on hosts without
//! AVX-512 — the ARM development machine included.

// The prototype has no production caller yet — it is measured through the
// `flock_probe_*` symbols and driven by its own tests — so nothing here is
// reachable from the prover on any target.
#![allow(dead_code)]

use crate::ntt::InvNttTableByteSingleGf8;

/// Rows consumed per batch. Free for the caller: a round-1 block is 32 rows of
/// 64 output bytes fed from `1 << N_MEDIUM = 16` `b_med` blocks of eight
/// K-rows, and the octa builder runs eight blocks per bout, so 64-row batches
/// fall out of the existing loop nest without restructuring.
pub(crate) const EXTEND_ROWS: usize = 64;

/// Packed input bytes per row (`n_chunks` at `k_skip = 6`).
pub(crate) const CHUNKS: usize = 8;

/// Output bytes per row (`ell` at `k_skip = 6`).
pub(crate) const ELL: usize = 64;

/// `FLOCK_NO_URM_GFNI_EXTEND=1` restores the table-gather inverse-NTT apply in
/// the shift-reduce AB path (exact same-binary A/B control). Resolved once per
/// process. The prototype is not yet wired into the production leaf; the gate
/// is present so the mechanism ships with the established kill switch.
pub(crate) fn urm_gfni_extend_enabled() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_NO_URM_GFNI_EXTEND").is_none());
    *ON
}

/// `FLOCK_GFNI_OCTA_ALL=1` gives the fused kernel all 32 medium windows,
/// including the four the static-B kernel is live for — the same-binary A/B of
/// the two reservation policies.
///
/// Opt-in, because the arithmetic genuinely cuts both ways and neither of the
/// two has been measured against the other: static-B skips the statically
/// known b side, so it runs roughly half the applies but still pays table
/// traffic, while the fused kernel runs both sides off 64 bytes of matrix
/// reads instead of 8 KiB of table. Default preserves static-B, which the
/// scoring machine has already paid for.
pub(crate) fn gfni_octa_all() -> bool {
    static ON: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var_os("FLOCK_GFNI_OCTA_ALL").is_some());
    *ON
}

// ---------------------------------------------------------------------------
// Matrix construction
// ---------------------------------------------------------------------------

/// The 64 `VGF2P8AFFINEQB` matrices of the §2.1 extension: `mats[i']` is the
/// 8×8 GF(2) matrix taking a packed input byte to `T₀[byte][i']`.
///
/// Encoding matches the hardware (`out.bit[i] = parity(mat.byte[7-i] & in)`),
/// built the same way as [`crate::lincheck`]'s `fold_mats_from_basis` and the
/// univariate-skip `build_one_group_mats`: gather byte-column `i'` of the
/// eight basis images into a qword, 8×8 bit-transpose it so byte `i` holds the
/// matrix row for output bit `i`, then `swap_bytes` to place row `i` at byte
/// `7 - i`.
///
/// The eight one-hot table rows `T₀[1 << t]` determine `T₀` completely by
/// GF(2)-linearity, so these 512 bytes replace the incumbent's 16 KiB table
/// (32 KiB with the σ₈ second image) in full.
pub(crate) fn build_extend_mats(inv_table: &InvNttTableByteSingleGf8) -> [u64; ELL] {
    assert_eq!(inv_table.ell, ELL, "prototype is specialized for ell = 64");
    assert_eq!(inv_table.n_chunks, CHUNKS, "prototype expects eight chunks");
    let base = inv_table.data_ptr();
    // cols[t] = T₀[1 << t] — the unit-column images of `fwd_NTT_Λ ∘ inv_NTT_S`.
    let cols: [&[u8]; 8] = std::array::from_fn(|t| {
        // SAFETY: the table holds 256 rows of `ell` readable bytes and
        // `1 << t <= 128`.
        unsafe { core::slice::from_raw_parts(base.add((1usize << t) * ELL), ELL) }
    });
    std::array::from_fn(|i_prime| {
        let mut col = 0u64;
        for (t, c) in cols.iter().enumerate() {
            col |= (c[i_prime] as u64) << (8 * t);
        }
        // `transpose_8x8_bits`: out.byte[i].bit[t] = in.byte[t].bit[i], i.e.
        // out.byte[i] is the matrix row for output bit `i`. GFNI wants that
        // row at byte `7 - i`.
        crate::bits::transpose_8x8_bits(col).swap_bytes()
    })
}

// ---------------------------------------------------------------------------
// 512-bit vector abstraction
// ---------------------------------------------------------------------------

/// 64-byte permutation index vector (`vpermb` control).
#[repr(C, align(64))]
pub(crate) struct Idx8(pub(crate) [u8; 64]);

/// 8-qword permutation index vector (`vpermt2q` control).
#[repr(C, align(64))]
pub(crate) struct Idx64(pub(crate) [u64; 8]);

/// The eight 512-bit instructions the kernel is built from. Implemented by
/// `__m512i` on x86-64 and by a portable byte-array model used as the
/// correctness oracle on hosts without AVX-512.
pub(crate) trait V512: Copy {
    fn zero() -> Self;

    /// `vmovdqu64 zmm, [p]`.
    ///
    /// # Safety
    /// `p` must point to 64 readable bytes.
    unsafe fn load(p: *const u8) -> Self;

    /// `vmovdqu64 [p], zmm`.
    ///
    /// # Safety
    /// `p` must point to 64 writable bytes.
    unsafe fn store(self, p: *mut u8);

    /// `vpxorq`.
    fn xor(self, o: Self) -> Self;

    /// `vpternlogq $0x96` — three-input XOR.
    fn xor3(self, b: Self, c: Self) -> Self;

    /// `vpermb`: `out.byte[i] = self.byte[idx[i] & 63]`.
    fn permb(self, idx: &Idx8) -> Self;

    /// `vpunpcklqdq`: per 128-bit lane `l`, `out.qword[2l] = self.qword[2l]`,
    /// `out.qword[2l+1] = o.qword[2l]`.
    fn unpacklo64(self, o: Self) -> Self;

    /// `vpunpckhqdq`: per 128-bit lane `l`, `out.qword[2l] = self.qword[2l+1]`,
    /// `out.qword[2l+1] = o.qword[2l+1]`.
    fn unpackhi64(self, o: Self) -> Self;

    /// `vpermt2q`: `out.qword[i]` is `self.qword[idx[i]]` for `idx[i] < 8` and
    /// `o.qword[idx[i] - 8]` otherwise.
    fn permt2q(self, idx: &Idx64, o: Self) -> Self;

    /// `vpbroadcastq`: one 8×8 matrix replicated to all eight qwords.
    fn mat_broadcast(mat: u64) -> Self;

    /// `vgf2p8affineqb`: per byte `x` of `self` and the matrix in the *same
    /// qword* of `mats`, `out.bit[i] = parity(mat.byte[7-i] & x)`.
    fn affine_v(self, mats: Self) -> Self;

    /// `vgf2p8affineqb` against one matrix shared by all 64 lanes — what the
    /// row-batched form needs, since all 64 rows take the same map.
    #[inline(always)]
    fn affine(self, mat: u64) -> Self {
        self.affine_v(Self::mat_broadcast(mat))
    }

    /// `vgf2p8mulb`: bytewise GF(2⁸) product under `x⁸+x⁴+x³+x+1` (0x11B).
    /// That is the AES polynomial, which is also
    /// [`crate::field::gf2_8`]'s — so the bilinear step of the shift-reduce
    /// needs no change of basis (had it differed, the change of basis is
    /// itself F₂-linear and would fold into the adjacent matrices for free).
    fn mulb(self, o: Self) -> Self;
}

/// Portable byte-array model of a ZMM register. Semantics of every [`V512`]
/// method match the corresponding AVX-512 instruction exactly; the tests use
/// it as the executable specification of the kernel on non-AVX-512 hosts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Bytes64(pub(crate) [u8; 64]);

impl Bytes64 {
    #[inline]
    fn qword(&self, i: usize) -> u64 {
        u64::from_le_bytes(self.0[i * 8..i * 8 + 8].try_into().unwrap())
    }

    #[inline]
    fn from_qwords(q: [u64; 8]) -> Self {
        let mut out = [0u8; 64];
        for (i, v) in q.iter().enumerate() {
            out[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        Bytes64(out)
    }
}

impl V512 for Bytes64 {
    #[inline]
    fn zero() -> Self {
        Bytes64([0u8; 64])
    }

    #[inline]
    unsafe fn load(p: *const u8) -> Self {
        let mut v = [0u8; 64];
        // SAFETY: caller guarantees 64 readable bytes; `v` is 64 writable
        // bytes and the regions cannot overlap (`v` is a fresh local).
        unsafe { core::ptr::copy_nonoverlapping(p, v.as_mut_ptr(), 64) };
        Bytes64(v)
    }

    #[inline]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: caller guarantees 64 writable bytes.
        unsafe { core::ptr::copy_nonoverlapping(self.0.as_ptr(), p, 64) };
    }

    #[inline]
    fn xor(self, o: Self) -> Self {
        Bytes64(std::array::from_fn(|i| self.0[i] ^ o.0[i]))
    }

    #[inline]
    fn xor3(self, b: Self, c: Self) -> Self {
        Bytes64(std::array::from_fn(|i| self.0[i] ^ b.0[i] ^ c.0[i]))
    }

    #[inline]
    fn permb(self, idx: &Idx8) -> Self {
        Bytes64(std::array::from_fn(|i| self.0[(idx.0[i] & 63) as usize]))
    }

    #[inline]
    fn unpacklo64(self, o: Self) -> Self {
        Self::from_qwords(std::array::from_fn(|i| {
            let lane = i / 2;
            if i % 2 == 0 {
                self.qword(2 * lane)
            } else {
                o.qword(2 * lane)
            }
        }))
    }

    #[inline]
    fn unpackhi64(self, o: Self) -> Self {
        Self::from_qwords(std::array::from_fn(|i| {
            let lane = i / 2;
            if i % 2 == 0 {
                self.qword(2 * lane + 1)
            } else {
                o.qword(2 * lane + 1)
            }
        }))
    }

    #[inline]
    fn permt2q(self, idx: &Idx64, o: Self) -> Self {
        Self::from_qwords(std::array::from_fn(|i| {
            let s = (idx.0[i] & 0xf) as usize;
            if s < 8 { self.qword(s) } else { o.qword(s - 8) }
        }))
    }

    #[inline]
    fn mat_broadcast(mat: u64) -> Self {
        Self::from_qwords([mat; 8])
    }

    #[inline]
    fn affine_v(self, mats: Self) -> Self {
        Bytes64(std::array::from_fn(|i| {
            // One matrix per qword — the hardware's actual granularity.
            let rows = mats.qword(i / 8).to_le_bytes();
            let x = self.0[i];
            let mut out = 0u8;
            for bit in 0..8 {
                // GFNI stores output row `bit` at byte `7 - bit`.
                if (rows[7 - bit] & x).count_ones() & 1 == 1 {
                    out |= 1 << bit;
                }
            }
            out
        }))
    }

    #[inline]
    fn mulb(self, o: Self) -> Self {
        // `F8`'s multiplication is GF(2⁸) mod 0x11B — the same polynomial
        // `VGF2P8MULB` implements.
        Bytes64(std::array::from_fn(|i| {
            (crate::field::F8(self.0[i]) * crate::field::F8(o.0[i])).0
        }))
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
impl V512 for core::arch::x86_64::__m512i {
    #[inline(always)]
    fn zero() -> Self {
        // SAFETY: avx512f is enabled by the impl's cfg gate.
        unsafe { core::arch::x86_64::_mm512_setzero_si512() }
    }

    #[inline(always)]
    unsafe fn load(p: *const u8) -> Self {
        // SAFETY: caller guarantees 64 readable bytes; avx512f per the gate.
        unsafe { core::arch::x86_64::_mm512_loadu_si512(p as *const core::arch::x86_64::__m512i) }
    }

    #[inline(always)]
    unsafe fn store(self, p: *mut u8) {
        // SAFETY: caller guarantees 64 writable bytes; avx512f per the gate.
        unsafe {
            core::arch::x86_64::_mm512_storeu_si512(p as *mut core::arch::x86_64::__m512i, self)
        };
    }

    #[inline(always)]
    fn xor(self, o: Self) -> Self {
        // SAFETY: avx512f is enabled by the impl's cfg gate.
        unsafe { core::arch::x86_64::_mm512_xor_si512(self, o) }
    }

    #[inline(always)]
    fn xor3(self, b: Self, c: Self) -> Self {
        // SAFETY: avx512f is enabled by the impl's cfg gate.
        unsafe { core::arch::x86_64::_mm512_ternarylogic_epi64::<0x96>(self, b, c) }
    }

    #[inline(always)]
    fn permb(self, idx: &Idx8) -> Self {
        // SAFETY: avx512bw/vbmi per the gate; `Idx8` is 64-byte aligned.
        unsafe {
            let iv = core::arch::x86_64::_mm512_load_si512(
                idx.0.as_ptr() as *const core::arch::x86_64::__m512i
            );
            core::arch::x86_64::_mm512_permutexvar_epi8(iv, self)
        }
    }

    #[inline(always)]
    fn unpacklo64(self, o: Self) -> Self {
        // SAFETY: avx512f is enabled by the impl's cfg gate.
        unsafe { core::arch::x86_64::_mm512_unpacklo_epi64(self, o) }
    }

    #[inline(always)]
    fn unpackhi64(self, o: Self) -> Self {
        // SAFETY: avx512f is enabled by the impl's cfg gate.
        unsafe { core::arch::x86_64::_mm512_unpackhi_epi64(self, o) }
    }

    #[inline(always)]
    fn permt2q(self, idx: &Idx64, o: Self) -> Self {
        // SAFETY: avx512f per the gate; `Idx64` is 64-byte aligned.
        unsafe {
            let iv = core::arch::x86_64::_mm512_load_si512(
                idx.0.as_ptr() as *const core::arch::x86_64::__m512i
            );
            core::arch::x86_64::_mm512_permutex2var_epi64(self, iv, o)
        }
    }

    #[inline(always)]
    fn mat_broadcast(mat: u64) -> Self {
        // SAFETY: avx512f is enabled by the impl's cfg gate.
        unsafe { core::arch::x86_64::_mm512_set1_epi64(mat as i64) }
    }

    #[inline(always)]
    fn affine_v(self, mats: Self) -> Self {
        // SAFETY: gfni + avx512f are enabled by the impl's cfg gate.
        unsafe { core::arch::x86_64::_mm512_gf2p8affine_epi64_epi8::<0>(self, mats) }
    }

    #[inline(always)]
    fn mulb(self, o: Self) -> Self {
        // SAFETY: gfni + avx512bw are enabled by the impl's cfg gate.
        unsafe { core::arch::x86_64::_mm512_gf2p8mul_epi8(self, o) }
    }
}

// ---------------------------------------------------------------------------
// Shuffle network (identical to `gfni_fold64_regs_impl`'s)
// ---------------------------------------------------------------------------

/// 8×8 byte transpose inside one ZMM: `out.qword[j] = { byte j of the ZMM's
/// eight qwords }`. `BT[8j + q] = 8q + j`, and the permutation is an
/// involution, so the same control drives the inverse.
#[rustfmt::skip]
static BT: Idx8 = Idx8([
    0, 8, 16, 24, 32, 40, 48, 56,   1, 9, 17, 25, 33, 41, 49, 57,
    2, 10, 18, 26, 34, 42, 50, 58,  3, 11, 19, 27, 35, 43, 51, 59,
    4, 12, 20, 28, 36, 44, 52, 60,  5, 13, 21, 29, 37, 45, 53, 61,
    6, 14, 22, 30, 38, 46, 54, 62,  7, 15, 23, 31, 39, 47, 55, 63,
]);

static S2_LO: Idx64 = Idx64([0, 1, 8, 9, 2, 3, 10, 11]);
static S2_HI: Idx64 = Idx64([4, 5, 12, 13, 6, 7, 14, 15]);
static S3_LO: Idx64 = Idx64([0, 1, 2, 3, 8, 9, 10, 11]);
static S3_HI: Idx64 = Idx64([4, 5, 6, 7, 12, 13, 14, 15]);
/// Last stage of [`fold_k`]: pair qword `j` with `j + 2` inside each half.
static R3_LO: Idx64 = Idx64([0, 1, 4, 5, 8, 9, 12, 13]);
static R3_HI: Idx64 = Idx64([2, 3, 6, 7, 10, 11, 14, 15]);

/// `out[j].qword[i] = t[i].qword[j]` — 8 `vpunpck` + 16 `vpermt2q`, the
/// network `gfni_fold64_regs_impl` uses in both directions (it is an
/// involution on the qword index pair).
#[inline(always)]
fn qword_transpose<V: V512>(t: [V; 8]) -> [V; 8] {
    let e01 = t[0].unpacklo64(t[1]);
    let o01 = t[0].unpackhi64(t[1]);
    let e23 = t[2].unpacklo64(t[3]);
    let o23 = t[2].unpackhi64(t[3]);
    let e45 = t[4].unpacklo64(t[5]);
    let o45 = t[4].unpackhi64(t[5]);
    let e67 = t[6].unpacklo64(t[7]);
    let o67 = t[6].unpackhi64(t[7]);
    let h02_a = e01.permt2q(&S2_LO, e23);
    let h46_a = e01.permt2q(&S2_HI, e23);
    let h13_a = o01.permt2q(&S2_LO, o23);
    let h57_a = o01.permt2q(&S2_HI, o23);
    let h02_b = e45.permt2q(&S2_LO, e67);
    let h46_b = e45.permt2q(&S2_HI, e67);
    let h13_b = o45.permt2q(&S2_LO, o67);
    let h57_b = o45.permt2q(&S2_HI, o67);
    [
        h02_a.permt2q(&S3_LO, h02_b),
        h13_a.permt2q(&S3_LO, h13_b),
        h02_a.permt2q(&S3_HI, h02_b),
        h13_a.permt2q(&S3_HI, h13_b),
        h46_a.permt2q(&S3_LO, h46_b),
        h57_a.permt2q(&S3_LO, h57_b),
        h46_a.permt2q(&S3_HI, h46_b),
        h57_a.permt2q(&S3_HI, h57_b),
    ]
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

/// 512 contiguous input bytes (row `r` at `src[8r..8r+8]`) → eight chunk
/// planes with `p[b].byte[r] = src[8r + b]`.
///
/// # Safety
/// `src` must point to `EXTEND_ROWS * CHUNKS` readable bytes.
#[inline(always)]
unsafe fn input_planes<V: V512>(src: *const u8) -> [V; 8] {
    // SAFETY: forwarded contract; the eight loads cover exactly 512 bytes.
    let t: [V; 8] = std::array::from_fn(|m| unsafe { V::load(src.add(64 * m)) }.permb(&BT));
    qword_transpose(t)
}

/// One output-byte plane: `out[i] = ⊕_b N[i ⊕ 8b] · p[b]`.
///
/// The `i ⊕ 8b` coordinate permutation of the §2.1 collapse is entirely a
/// *matrix index* here — it costs no instruction. Eight affine products folded
/// by three `vpternlogq` and one `vpxorq`.
///
/// Only the *shape* of the schedule: kept as documentation of the plain form
/// and as the fallback the `v`-slice schedule below is checked against.
#[inline(always)]
#[cfg(test)]
fn out_plane<V: V512>(p: &[V; 8], mats: &[u64; ELL], i: usize) -> V {
    let g = |b: usize| p[b].affine(mats[i ^ (8 * b)]);
    let v1 = g(0).xor3(g(1), g(2));
    let v2 = g(3).xor3(g(4), g(5));
    let v3 = g(6).xor3(g(7), v1);
    v2.xor(v3)
}

/// One output-byte plane in the `v`-slice schedule.
///
/// Writing the output index as `i = 8u + v` and substituting `w = u ⊕ b`,
///
/// ```text
///   out[8u + v] = ⊕_b N[8(u ⊕ b) + v] · p[b] = ⊕_w M_v[w] · p[u ⊕ w]
/// ```
///
/// with `M_v[w] = N[8w + v]`. The eight matrices of a `v` slice therefore
/// serve **all eight** of its output planes, so they are broadcast into
/// registers once per slice (`mv`) and the inner eight planes touch no memory
/// at all — only the plane index `u ⊕ w` moves, and it is a constant register
/// selection once `u` is unrolled.
///
/// This matters for codegen, not just elegance: indexing `mats[i ^ 8b]` from a
/// rolled loop costs a `mov` + `xor` GPR pair per term (eight per plane), and
/// fully unrolling all 64 planes instead makes LLVM hoist all 512 broadcasts
/// and spill ~13 KiB of stack. Both measured; both avoided here.
#[inline(always)]
fn out_plane_v<V: V512>(p: &[V; 8], mv: &[V; 8], u: usize) -> V {
    let g = |w: usize| p[u ^ w].affine_v(mv[w]);
    let v1 = g(0).xor3(g(1), g(2));
    let v2 = g(3).xor3(g(4), g(5));
    let v3 = g(6).xor3(g(7), v1);
    v2.xor(v3)
}

/// Expand `$body!(u)` once for each of the eight planes of a `v` slice.
macro_rules! each_u {
    ($body:ident) => {
        $body!(0); $body!(1); $body!(2); $body!(3);
        $body!(4); $body!(5); $body!(6); $body!(7);
    };
}

/// Extend 64 rows into **plane-major** output: `out[64*i + r] = apply(row
/// r)[i]`. This is the form a fused a·b pipeline consumes — `VGF2P8MULB` and
/// the `x^k` shift-reduce are pointwise in `(row, lane)` and so are indifferent
/// to which of the two layouts they see, and a fused kernel would keep these
/// planes in registers rather than storing them.
///
/// # Safety
/// `src`: `EXTEND_ROWS * CHUNKS` readable bytes. `out`: `EXTEND_ROWS * ELL`
/// writable bytes.
#[inline(always)]
pub(crate) unsafe fn extend64_planes<V: V512>(src: *const u8, mats: &[u64; ELL], out: *mut u8) {
    // SAFETY: forwarded contract.
    unsafe {
        let p = input_planes::<V>(src);
        for v in 0..8 {
            let mv: [V; 8] = std::array::from_fn(|w| V::mat_broadcast(mats[8 * w + v]));
            macro_rules! emit {
                ($u:literal) => {
                    out_plane_v(&p, &mv, $u).store(out.add(ELL * (8 * $u + v)))
                };
            }
            each_u!(emit);
        }
    }
}

/// Extend 64 rows into **row-major** output: `out[64*r + i] = apply(row r)[i]`
/// — a drop-in replacement for 64 calls of
/// [`InvNttTableByteSingleGf8::apply`].
///
/// The tail is a 64×64 byte transpose, run as the inverse of the input network
/// in two stages: per group of eight output-byte planes, `qword_transpose` +
/// `vpermb` gives, for each `m`, the eight rows `8m..8m+8` restricted to that
/// group's eight output bytes; a second `qword_transpose` per `m` then
/// interleaves the eight groups into whole rows. 448 port-5 shuffles per 64
/// rows plus the 64-vector spill the register file cannot hold.
///
/// # Safety
/// `src`: `EXTEND_ROWS * CHUNKS` readable bytes. `out`: `EXTEND_ROWS * ELL`
/// writable bytes.
#[inline(always)]
pub(crate) unsafe fn extend64_rows<V: V512>(src: *const u8, mats: &[u64; ELL], out: *mut u8) {
    // SAFETY: forwarded contract.
    unsafe {
        let p = input_planes::<V>(src);
        // 64 output-byte planes is twice the register file, so this buffer is a
        // real spill and is counted: 64 stores plus 64 reloads on top of the
        // 448 shuffles. `MaybeUninit` because every slot is written before it
        // is read and a `[V::zero(); 64]` seed costs two 4 KiB `memset`s.
        let mut q: [core::mem::MaybeUninit<V>; ELL] = [const { core::mem::MaybeUninit::uninit() }; ELL];
        for v in 0..8 {
            let mv: [V; 8] = std::array::from_fn(|w| V::mat_broadcast(mats[8 * w + v]));
            macro_rules! emit {
                ($u:literal) => {
                    q[8 * $u + v].write(out_plane_v(&p, &mv, $u))
                };
            }
            each_u!(emit);
        }
        // In place: `q[8g + m].qword[a]` becomes output bytes `8g..8g+8` of row
        // `8m + a`. One buffer, not two.
        for g in 0..8 {
            let r = qword_transpose(std::array::from_fn(|j| q[8 * g + j].assume_init_read()));
            for (m, v) in r.into_iter().enumerate() {
                q[8 * g + m].write(v.permb(&BT));
            }
        }
        for m in 0..8 {
            let rows =
                qword_transpose(std::array::from_fn(|g| q[8 * g + m].assume_init_read()));
            for (a, v) in rows.into_iter().enumerate() {
                v.store(out.add((8 * m + a) * ELL));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused a·b round-1 projection, plane-major
// ---------------------------------------------------------------------------

/// Rows are `r = 8·blk + k` in memory (window `blk`'s eight K-rows are eight
/// consecutive packed bytes each). `permb(BT)` on a plane swaps the two index
/// halves, giving byte position `8k + blk` — **k-major**, so every qword holds
/// a single `k`.
///
/// Free: the affine battery treats all 64 byte positions identically (one
/// matrix broadcast to all lanes), so relabeling the eight *input* planes
/// relabels all 64 output planes with it. Eight `vpermb` per operand, not per
/// output.
///
/// # Safety
/// `src` must point to `EXTEND_ROWS * CHUNKS` readable bytes.
#[inline(always)]
unsafe fn input_planes_kmajor<V: V512>(src: *const u8) -> [V; 8] {
    // SAFETY: forwarded contract.
    let p = unsafe { input_planes::<V>(src) };
    p.map(|v| v.permb(&BT))
}

/// `x^k` as a bytewise multiplier in the k-major layout: byte `8k + blk` gets
/// `x^k = 1 << k`. One `vgf2p8mulb` per output plane replaces the incumbent's
/// per-K `vpbroadcastb` + `vgf2p8mulb` pair, and needs no per-qword matrix.
#[inline(always)]
fn xk_vector<V: V512>() -> V {
    let mut bytes = [0u8; 64];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = 1u8 << (i / 8);
    }
    // SAFETY: `bytes` is exactly 64 readable bytes.
    unsafe { V::load(bytes.as_ptr()) }
}

/// XOR of eight vectors in four `vpternlogq`/`vpxorq` ops.
#[inline(always)]
fn xor8<V: V512>(t: &[V; 8]) -> V {
    let v1 = t[0].xor3(t[1], t[2]);
    let v2 = t[3].xor3(t[4], t[5]);
    let v3 = t[6].xor3(t[7], v1);
    v2.xor(v3)
}

/// [`build_extend_mats`] in the v-slice layout the fused kernel wants:
/// `out[v][w] = N[8w + v]`.
///
/// Not cosmetic. Indexing the flat block as `mats[8*w + v]` from the `v` loop
/// leaves LLVM unable to prove `8w + v < 64` and it emits a bounds check plus a
/// `panic_bounds_check` edge inside the battery (measured). With the slice
/// materialized, `w` is a literal into a `[u64; 8]` and the eight matrices of a
/// slice are contiguous, which is also what lets them reach `vgf2p8affineqb`
/// as EVEX embedded-broadcast memory operands.
pub(crate) fn build_extend_mats_vslice(inv_table: &InvNttTableByteSingleGf8) -> [[u64; 8]; 8] {
    let flat = build_extend_mats(inv_table);
    std::array::from_fn(|v| std::array::from_fn(|w| flat[8 * w + v]))
}

// --- the k fold -----------------------------------------------------------
//
// Given eight scaled planes `s[u]`, produce one vector with
// `out.qword[u] = ⊕_m s[u].qword[m]` — the `x^k` sum of the slice's eight
// output-byte planes, since in the k-major layout a plane's qword `m` is K-row
// `m`. Naively that is `qword_transpose(s)` (24 shuffles) plus a four-op
// eight-way XOR = 28 ops with all eight planes live at once, which spilled
// (measured: 208 stack stores and 254 reloads per eight windows).
//
// The butterfly below halves the vector count at every stage instead, so it
// costs 21 ops (14 shuffles + 7 XORs) AND consumes each pair of planes as soon
// as it is produced — at most two `s` planes and four partials are ever live.

/// Stage 1. `out.qword[2l + e] = s_e.qword[2l] ^ s_e.qword[2l+1]` — the two
/// planes' `m`-pairs, interleaved by the low bit.
#[inline(always)]
fn fold_pair<V: V512>(a: V, b: V) -> V {
    a.unpacklo64(b).xor(a.unpackhi64(b))
}

/// Stage 2. Halves the `m` axis again, packing two stage-1 vectors into one.
#[inline(always)]
fn fold_join<V: V512>(a: V, b: V) -> V {
    a.permt2q(&S3_LO, b).xor(a.permt2q(&S3_HI, b))
}

/// Stage 3. Last `m` halving; lands plane `u`'s total in qword `u`.
#[inline(always)]
fn fold_final<V: V512>(a: V, b: V) -> V {
    a.permt2q(&R3_LO, b).xor(a.permt2q(&R3_HI, b))
}

/// Reference form of the fold, kept as the test oracle for the butterfly.
#[inline(always)]
#[cfg(test)]
fn fold_k_reference<V: V512>(s: [V; 8]) -> V {
    xor8(&qword_transpose(s))
}

/// One fused output-byte plane: extend a, extend b, multiply, scale by `x^k`.
///
/// A free-standing `#[inline(always)]` function rather than a closure — as a
/// closure LLVM declined to inline it and emitted eight real `callq`s per
/// slice (measured).
#[inline(always)]
fn fused_plane<V: V512>(pa: &[V; 8], pb: &[V; 8], mv: &[u64; 8], xk: V, u: usize) -> V {
    // Accumulator chain, not a balanced tree. Both spend three `vpternlogq`
    // and one `vpxorq`, but the tree holds two partial sums live at once and
    // the chain holds one — and with sixteen input planes plus eight
    // broadcast matrices already resident, those spare registers are the
    // difference between a clean slice and a spilling one. Latency does not
    // matter here: the slice's eight planes are independent, so the machine
    // has plenty to overlap.
    let battery = |p: &[V; 8]| {
        let g = |w: usize| p[u ^ w].affine(mv[w]);
        let acc = g(0).xor3(g(1), g(2));
        let acc = acc.xor3(g(3), g(4));
        let acc = acc.xor3(g(5), g(6));
        acc.xor(g(7))
    };
    battery(pa).mulb(battery(pb)).mulb(xk)
}

/// **The fused pipeline.** Eight 64-byte round-1 medium windows in one call:
/// `a_src` and `b_src` are the 8 × 64 contiguous packed bytes that
/// `StreamProj`'s staging already holds for one drain step, and the result is
/// window `j`'s 64 transformed bytes in `out[j]`.
///
/// Bit-identical to eight `shift_reduce_inner_ab_at` calls. The pipeline is
///
/// 1. eight chunk planes per operand, k-major (`input_planes_kmajor`);
/// 2. per `v` slice, per `u`: the a and b output-byte planes
///    `⊕_w N[8w+v]·p[u⊕w]` — **the same 64 matrices serve both operands**, so
///    the a and b batteries share every matrix operand;
/// 3. `vgf2p8mulb` for the bilinear step — no basis change (see [`V512::mulb`]);
/// 4. `vgf2p8mulb` by [`xk_vector`] for the `x^k` scaling;
/// 5. the `k` fold, batched eight planes at a time: `qword_transpose` turns
///    "XOR the eight qwords of each plane" into "XOR eight vectors", which is
///    where the plane-major form pays its only new shuffles;
/// 6. the `(vector, low-3-bits-of-byte)` swap that turns the eight folded `v`
///    slices back into eight whole windows.
///
/// Nothing is ever materialized row-major: steps 3-5 are pointwise in
/// `(row, lane)` and indifferent to layout, so the 64×64 transpose that made
/// the standalone row-major kernel a 1.19x non-starter never happens.
///
/// # Safety
/// `a_src`, `b_src`: `EXTEND_ROWS * CHUNKS` readable bytes each.
#[inline(always)]
pub(crate) unsafe fn fused_ab8<V: V512>(
    a_src: *const u8,
    b_src: *const u8,
    mats: &[[u64; 8]; 8],
) -> [V; 8] {
    // SAFETY: forwarded contract.
    unsafe {
        let pa = input_planes_kmajor::<V>(a_src);
        let pb = input_planes_kmajor::<V>(b_src);
        let xk = xk_vector::<V>();

        // `f[v].qword[u].byte[blk]` = output byte `8u + v` of window `blk`.
        let mut f: [core::mem::MaybeUninit<V>; 8] =
            [const { core::mem::MaybeUninit::uninit() }; 8];
        for (slot, mv) in f.iter_mut().zip(mats.iter()) {
            // The matrices reach `vgf2p8affineqb` as EVEX embedded-broadcast
            // memory operands off the 512-byte block, so they cost neither a
            // register nor a `vpbroadcastq`.
            // Each pair of planes is folded the moment it exists, so the eight
            // never coexist — see the fold butterfly above.
            let mut p2 = |u: usize| {
                fold_pair(
                    fused_plane(&pa, &pb, mv, xk, u),
                    fused_plane(&pa, &pb, mv, xk, u + 1),
                )
            };
            let (y0, y1, y2, y3) = (p2(0), p2(2), p2(4), p2(6));
            slot.write(fold_final(fold_join(y0, y1), fold_join(y2, y3)));
        }

        // `f[v].byte[8u + blk]` → window `blk`, output byte `8u + v`: swap the
        // vector index with the low three bits of the byte index.
        let g: [V; 8] = std::array::from_fn(|v| f[v].assume_init_read().permb(&BT));
        qword_transpose(g).map(|h| h.permb(&BT))
    }
}

/// Portable instantiation of [`fused_ab8`] — the executable specification.
///
/// # Safety
/// As [`fused_ab8`].
#[inline]
pub(crate) unsafe fn fused_ab8_model(
    a_src: *const u8,
    b_src: *const u8,
    mats: &[[u64; 8]; 8],
    out: &mut [[u8; ELL]; 8],
) {
    // SAFETY: forwarded contract; each `out[j]` is 64 writable bytes.
    unsafe {
        let r = fused_ab8::<Bytes64>(a_src, b_src, mats);
        for (dst, v) in out.iter_mut().zip(r) {
            *dst = v.0;
        }
    }
}

/// Portable instantiation — the executable specification. Same source as the
/// AVX-512 kernel, run through the [`Bytes64`] instruction model.
///
/// # Safety
/// As [`extend64_rows`].
#[inline]
pub(crate) unsafe fn extend64_rows_model(src: *const u8, mats: &[u64; ELL], out: *mut u8) {
    // SAFETY: forwarded contract.
    unsafe { extend64_rows::<Bytes64>(src, mats, out) }
}

/// Portable instantiation of [`extend64_planes`].
///
/// # Safety
/// As [`extend64_planes`].
#[inline]
pub(crate) unsafe fn extend64_planes_model(src: *const u8, mats: &[u64; ELL], out: *mut u8) {
    // SAFETY: forwarded contract.
    unsafe { extend64_planes::<Bytes64>(src, mats, out) }
}

// ---------------------------------------------------------------------------
// Static instruction-count probes (SPR assembly)
// ---------------------------------------------------------------------------

/// Source block for the probes: 64 rows × 8 packed bytes.
pub type ExtendSrc = [u8; EXTEND_ROWS * CHUNKS];
/// Destination block for the probes: 64 rows × 64 output bytes.
pub type ExtendDst = [u8; EXTEND_ROWS * ELL];

/// Asm probe: row-major GFNI extension of 64 rows. `#[unsafe(no_mangle)]` so
/// the symbol survives to the emitted assembly for static instruction and
/// port-5 accounting; not called from the prover.
///
/// All three probes take the same `&`/`&mut` shapes so `noalias` holds equally
/// on both sides of the comparison and neither gets an unfair hoisting
/// advantage.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[unsafe(no_mangle)]
pub extern "C" fn flock_probe_gfni_extend64_rows(
    src: &ExtendSrc,
    mats: &[u64; ELL],
    out: &mut ExtendDst,
) {
    // SAFETY: the array types carry exactly the byte counts the kernel reads
    // and writes.
    unsafe { extend64_rows::<core::arch::x86_64::__m512i>(src.as_ptr(), mats, out.as_mut_ptr()) }
}

/// Asm probe: plane-major GFNI extension of 64 rows (no output transpose).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[unsafe(no_mangle)]
pub extern "C" fn flock_probe_gfni_extend64_planes(
    src: &ExtendSrc,
    mats: &[u64; ELL],
    out: &mut ExtendDst,
) {
    // SAFETY: as above.
    unsafe { extend64_planes::<core::arch::x86_64::__m512i>(src.as_ptr(), mats, out.as_mut_ptr()) }
}

// ---------------------------------------------------------------------------
// Resolution and the production entry point
// ---------------------------------------------------------------------------

/// Process-wide cache of the v-slice matrix block together with a fingerprint
/// of the table it came from. The eight one-hot rows `T₀[1 << t]` determine
/// `T₀` completely by GF(2)-linearity — that is exactly what
/// [`build_extend_mats`] reads — so they are the fingerprint. Same scheme as
/// `x86_64_bstatic::prepare_bstatic`.
struct CachedMats {
    fingerprint: [u8; 8 * ELL],
    mats: [[u64; 8]; 8],
}

static MATS_CACHE: std::sync::OnceLock<CachedMats> = std::sync::OnceLock::new();

fn fingerprint_of(inv_table: &InvNttTableByteSingleGf8) -> [u8; 8 * ELL] {
    let mut fp = [0u8; 8 * ELL];
    let base = inv_table.data_ptr();
    for t in 0..8 {
        // SAFETY: the table has 256 rows of `ell == 64` readable bytes.
        let row = unsafe { core::slice::from_raw_parts(base.add((1usize << t) * ELL), ELL) };
        fp[t * ELL..(t + 1) * ELL].copy_from_slice(row);
    }
    fp
}

/// Resolve the fused kernel's 512-byte matrix block for `inv_table`, or `None`
/// when the mechanism is off, the kernel does not exist on this target, or the
/// table is not the protocol-fixed `k = 6` shape the matrices describe. The
/// caller then runs the incumbent leaf, which is always correct.
///
/// Called once per output buffer from `prepare_round1_ab_window_plan`, never
/// per window.
pub(crate) fn prepare_extend_mats(
    inv_table: &InvNttTableByteSingleGf8,
) -> Option<&'static [[u64; 8]; 8]> {
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    )))]
    {
        let _ = inv_table;
        return None;
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx512f",
        target_feature = "avx512bw",
        target_feature = "avx512vbmi",
        target_feature = "gfni"
    ))]
    {
        if !urm_gfni_extend_enabled()
            || inv_table.k != 6
            || inv_table.ell != ELL
            || inv_table.n_chunks != CHUNKS
        {
            return None;
        }
        let cached = MATS_CACHE.get_or_init(|| CachedMats {
            fingerprint: fingerprint_of(inv_table),
            mats: build_extend_mats_vslice(inv_table),
        });
        (cached.fingerprint == fingerprint_of(inv_table)).then_some(&cached.mats)
    }
}

/// The octa batch plus its live-predicated scatter, shared verbatim by the
/// AVX-512 entry point and by the portable model the tests drive — so the
/// window→destination addressing is pinned on hosts without AVX-512, not
/// merely written twice and hoped to agree.
///
/// # Safety
/// `a_stage`/`b_stage`: `8 * 64` readable bytes each. `store` is invoked once
/// per live window with `out_base + j * out_stride`.
#[inline(always)]
unsafe fn octa_scatter<V: V512>(
    a_stage: *const u8,
    b_stage: *const u8,
    mats: &[[u64; 8]; 8],
    out_base: *mut u8,
    out_stride: usize,
    live: u32,
    mut store: impl FnMut(*mut u8, V),
) {
    // SAFETY: forwarded contract.
    unsafe {
        let r = fused_ab8::<V>(a_stage, b_stage, mats);
        for (j, v) in r.into_iter().enumerate() {
            if live & (1 << j) == 0 {
                continue;
            }
            store(out_base.add(j * out_stride), v);
        }
    }
}

/// Portable instantiation of [`octa_scatter`] — the executable specification
/// of the production entry point, addressing and live mask included.
///
/// # Safety
/// As [`octa_scatter`], plus 64 writable bytes at each live destination.
#[inline]
pub(crate) unsafe fn round1_ab_inner_octa_model(
    a_stage: *const u8,
    b_stage: *const u8,
    mats: &[[u64; 8]; 8],
    out_base: *mut u8,
    out_stride: usize,
    live: u32,
) {
    // SAFETY: forwarded contract.
    unsafe {
        octa_scatter::<Bytes64>(a_stage, b_stage, mats, out_base, out_stride, live, |p, v| {
            v.store(p)
        });
    }
}

/// Production entry: transform the eight medium windows a drain step staged,
/// publishing each live one under the caller's store policy.
///
/// `a_stage`/`b_stage` are the `8 × 64` contiguous packed bytes the streaming
/// producer already holds; window `j` occupies bytes `64j..64j+64` and lands
/// at `out_base + j * out_stride`. Bit `j` of `live` selects it. Dead windows
/// are computed and discarded rather than branched around — their staged bytes
/// are readable, and eight predicated stores cost less than splitting the
/// batch.
///
/// # Safety
/// `a_stage` and `b_stage` must each point to `8 * 64` readable bytes.
/// `out_base + j * out_stride` must be 64 writable bytes for every `j` with
/// bit `j` of `live` set, with the alignment `nt` was classified for.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[inline]
pub(crate) unsafe fn round1_ab_inner_octa_gfni(
    a_stage: *const u8,
    b_stage: *const u8,
    out_base: *mut u8,
    out_stride: usize,
    live: u32,
    mats: &[[u64; 8]; 8],
    nt: u8,
) {
    // SAFETY: forwarded contract; `store_out64` upholds the `nt` alignment
    // contract the plan classified.
    unsafe {
        octa_scatter::<core::arch::x86_64::__m512i>(
            a_stage,
            b_stage,
            mats,
            out_base,
            out_stride,
            live,
            |p, v| super::x86_64::store_out64(&mut *p.cast::<[u8; ELL]>(), v, nt),
        );
    }
}

/// Asm probe: the fused plane-major a·b projection of eight medium windows.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vbmi",
    target_feature = "gfni"
))]
#[unsafe(no_mangle)]
pub extern "C" fn flock_probe_gfni_fused_ab8(
    a_src: &ExtendSrc,
    b_src: &ExtendSrc,
    mats: &[[u64; 8]; 8],
    out: &mut [[u8; ELL]; 8],
) {
    use core::arch::x86_64::*;
    // SAFETY: the array types carry exactly the byte counts the kernel touches.
    unsafe {
        let r = fused_ab8::<__m512i>(a_src.as_ptr(), b_src.as_ptr(), mats);
        for (dst, v) in out.iter_mut().zip(r) {
            _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, v);
        }
    }
}

/// Asm probe: the **incumbent** fused leaf over the same eight windows — the
/// exact per-drain-step computation `StreamProj::project` performs, on the
/// current `_prepared` path with the three selectors already resolved (so the
/// baseline includes the `ShiftReducePlan` hoist, not the older form).
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "gfni"
))]
#[unsafe(no_mangle)]
pub extern "C" fn flock_probe_incumbent_ab8(
    a_src: &ExtendSrc,
    b_src: &ExtendSrc,
    inv_table: &InvNttTableByteSingleGf8,
    out: &mut [[u8; ELL]; 8],
) {
    // SAFETY: 64 readable bytes per window on each side; `out[j]` is one
    // writable ZMM. The probe is never called from the prover.
    unsafe {
        for (j, dst) in out.iter_mut().enumerate() {
            // Straight to the arm the production selectors resolve to
            // (`img2 && pidx`, `offw = true`), so the baseline is the taken
            // path with its three booleans folded — the same specialization
            // the GFNI side gets — rather than the multi-arm dispatcher.
            super::x86_64::shift_reduce_inner_ab_x86_avx512_pidx(
                &a_src[j * 64..j * 64 + 64],
                &b_src[j * 64..j * 64 + 64],
                inv_table,
                0,
                0,
                dst,
                0,
                true,
            );
        }
    }
}

/// Asm probe: the **incumbent** a-side extension over the same 64 rows —
/// the `pidx` byte→`u16 * 64` prologue of
/// `shift_reduce_inner_ab_x86_avx512_pidx` followed by 64 two-image table
/// applies, each storing its 64-byte result. Same work, same codegen settings,
/// so the symbols are directly comparable.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "gfni"
))]
#[unsafe(no_mangle)]
pub extern "C" fn flock_probe_incumbent_extend64(
    src: &ExtendSrc,
    inv_table: &InvNttTableByteSingleGf8,
    out: &mut ExtendDst,
) {
    use core::arch::x86_64::*;
    #[repr(align(64))]
    struct Off([u16; EXTEND_ROWS * CHUNKS]);
    // SAFETY: `off` is fully written before use and every entry is `byte * 64`,
    // inside a 256-row image of 64-byte rows; the array types carry the byte
    // counts. The caller must pass the `k = 6` table with the σ₈ second image
    // (the probe is never called from the prover).
    unsafe {
        let mut off = core::mem::MaybeUninit::<Off>::uninit();
        let op = core::ptr::addr_of_mut!((*off.as_mut_ptr()).0) as *mut u16;
        for w in 0..(EXTEND_ROWS * CHUNKS / 32) {
            let v = _mm512_slli_epi16::<6>(_mm512_cvtepu8_epi16(_mm256_loadu_si256(
                src.as_ptr().add(32 * w) as *const __m256i,
            )));
            _mm512_store_si512(op.add(32 * w) as *mut __m512i, v);
        }
        let dst = out.as_mut_ptr();
        for r in 0..EXTEND_ROWS {
            let v = inv_table.apply_x86_avx512_register_2img_offw_unchecked(op.add(r * CHUNKS));
            _mm512_storeu_si512(dst.add(r * ELL) as *mut __m512i, v);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::F8;
    use crate::ntt::AdditiveNttGf8;

    fn table() -> InvNttTableByteSingleGf8 {
        let ntt_s = AdditiveNttGf8::new(6, F8::ZERO);
        let ntt_l = AdditiveNttGf8::new(6, F8(1u8 << 6));
        InvNttTableByteSingleGf8::new(&ntt_s, &ntt_l)
    }

    /// xorshift64* — deterministic inputs without a dev-dependency.
    fn rng(state: &mut u64) -> u64 {
        let mut x = *state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        *state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn random_rows(seed: u64) -> [u8; EXTEND_ROWS * CHUNKS] {
        let mut s = seed | 1;
        std::array::from_fn(|_| rng(&mut s) as u8)
    }

    /// 64 × `apply` — the oracle.
    fn oracle(t: &InvNttTableByteSingleGf8, src: &[u8; EXTEND_ROWS * CHUNKS]) -> Vec<u8> {
        let mut out = vec![0u8; EXTEND_ROWS * ELL];
        let mut col = [F8::ZERO; ELL];
        for r in 0..EXTEND_ROWS {
            t.apply(&src[r * CHUNKS..r * CHUNKS + CHUNKS], &mut col);
            for i in 0..ELL {
                out[r * ELL + i] = col[i].0;
            }
        }
        out
    }

    /// The 64 matrices reproduce every one of the 256 × 64 table entries:
    /// `mats[i'] · w == T₀[w][i']` for all `w`, `i'`. This is the whole
    /// GF(2)-linearity claim, checked exhaustively rather than sampled.
    #[test]
    fn mats_reproduce_the_whole_table() {
        let t = table();
        let mats = build_extend_mats(&t);
        // SAFETY: the table holds 256 rows of 64 readable bytes.
        let data = unsafe { core::slice::from_raw_parts(t.data_ptr(), 256 * ELL) };
        for w in 0..256usize {
            let broadcast = Bytes64([w as u8; 64]);
            for i_prime in 0..ELL {
                let got = broadcast.affine(mats[i_prime]).0[0];
                assert_eq!(got, data[w * ELL + i_prime], "w={w} i'={i_prime}");
            }
        }
    }

    #[test]
    fn extend64_rows_matches_apply() {
        let t = table();
        let mats = build_extend_mats(&t);
        for seed in [1u64, 0xC0FF_EE42, 0xDEAD_BEEF_1234_5678] {
            let src = random_rows(seed);
            let want = oracle(&t, &src);
            let mut got = vec![0u8; EXTEND_ROWS * ELL];
            // SAFETY: `src` is 512 bytes, `got` is 4096 bytes.
            unsafe { extend64_rows_model(src.as_ptr(), &mats, got.as_mut_ptr()) };
            assert_eq!(got, want, "seed={seed:#x}");
        }
    }

    /// Plane-major output is the same values under the `(r, i) -> (i, r)`
    /// relabeling, so the transpose-free kernel is exactly as correct as the
    /// row-major one.
    #[test]
    fn extend64_planes_is_the_transpose_of_rows() {
        let t = table();
        let mats = build_extend_mats(&t);
        let src = random_rows(0x5EED_1234);
        let want = oracle(&t, &src);
        let mut planes = vec![0u8; EXTEND_ROWS * ELL];
        // SAFETY: `src` is 512 bytes, `planes` is 4096 bytes.
        unsafe { extend64_planes_model(src.as_ptr(), &mats, planes.as_mut_ptr()) };
        for r in 0..EXTEND_ROWS {
            for i in 0..ELL {
                assert_eq!(planes[i * ELL + r], want[r * ELL + i], "r={r} i={i}");
            }
        }
    }

    /// Negative control: perturbing a single bit of a single matrix must make
    /// the oracle comparison fail. Without this the test above would pass for
    /// a kernel that ignored its matrices.
    #[test]
    fn perturbed_matrix_fails_the_oracle() {
        let t = table();
        let good = build_extend_mats(&t);
        let src = random_rows(0xA5A5_5A5A);
        let want = oracle(&t, &src);

        let mut caught = 0usize;
        for (idx, bit) in [(0usize, 0u32), (17, 31), (63, 63)] {
            let mut bad = good;
            bad[idx] ^= 1u64 << bit;
            let mut got = vec![0u8; EXTEND_ROWS * ELL];
            // SAFETY: `src` is 512 bytes, `got` is 4096 bytes.
            unsafe { extend64_rows_model(src.as_ptr(), &bad, got.as_mut_ptr()) };
            assert_ne!(
                got, want,
                "flipping bit {bit} of mats[{idx}] left the output unchanged"
            );
            caught += 1;
        }
        assert_eq!(caught, 3);

        // And the unperturbed matrices still agree — the control is sharp,
        // not a blanket mismatch.
        let mut got = vec![0u8; EXTEND_ROWS * ELL];
        // SAFETY: as above.
        unsafe { extend64_rows_model(src.as_ptr(), &good, got.as_mut_ptr()) };
        assert_eq!(got, want);
    }

    /// The `v`-slice schedule (`out_plane_v`, `i = 8u + v`, `w = u ⊕ b`) and
    /// the plain schedule (`out_plane`, `⊕_b N[i ⊕ 8b]`) must agree plane for
    /// plane — the reindexing is the only thing separating the shipped
    /// register-resident form from the obvious one.
    #[test]
    fn v_slice_schedule_matches_the_plain_one() {
        let t = table();
        let mats = build_extend_mats(&t);
        let src = random_rows(0x1357_9BDF);
        // SAFETY: `src` is 512 readable bytes.
        let p = unsafe { input_planes::<Bytes64>(src.as_ptr()) };
        for v in 0..8 {
            let mv: [Bytes64; 8] =
                std::array::from_fn(|w| Bytes64::mat_broadcast(mats[8 * w + v]));
            for u in 0..8 {
                assert_eq!(
                    out_plane_v(&p, &mv, u),
                    out_plane(&p, &mats, 8 * u + v),
                    "u={u} v={v}"
                );
            }
        }
    }

    /// Scalar oracle for one 64-byte medium window: the shift-reduce leaf
    /// spelled out from `portable::shift_reduce_inner_ab_scalar`, which is the
    /// definition every architecture backend is checked against.
    fn window_oracle(t: &InvNttTableByteSingleGf8, a: &[u8], b: &[u8]) -> [u8; ELL] {
        let mut acc = [0u16; ELL];
        let mut a_col = [F8::ZERO; ELL];
        let mut b_col = [F8::ZERO; ELL];
        for k in 0..8 {
            t.apply(&a[k * CHUNKS..k * CHUNKS + CHUNKS], &mut a_col);
            t.apply(&b[k * CHUNKS..k * CHUNKS + CHUNKS], &mut b_col);
            for lane in 0..ELL {
                acc[lane] ^= ((a_col[lane] * b_col[lane]).0 as u16) << k;
            }
        }
        std::array::from_fn(|lane| crate::field::gf2_8::gf8_reduce(acc[lane]))
    }

    /// The whole fused pipeline against the leaf oracle: eight windows, the
    /// bilinear `vgf2p8mulb` step, the `x^k` scaling and the `k` fold
    /// included. This is the byte-identity claim.
    #[test]
    fn fused_ab8_matches_the_shift_reduce_leaf() {
        let t = table();
        let mats = build_extend_mats_vslice(&t);
        for seed in [3u64, 0x9E37_79B9_7F4A_7C15, 0x0BAD_C0DE] {
            let a = random_rows(seed);
            let b = random_rows(seed ^ 0xFFFF_FFFF);
            let mut got = [[0u8; ELL]; 8];
            // SAFETY: both sources are 512 readable bytes.
            unsafe { fused_ab8_model(a.as_ptr(), b.as_ptr(), &mats, &mut got) };
            for j in 0..8 {
                let want = window_oracle(&t, &a[j * 64..j * 64 + 64], &b[j * 64..j * 64 + 64]);
                assert_eq!(got[j], want, "seed={seed:#x} window={j}");
            }
        }
    }

    /// Negative control for the fused path: the `x^k` scaling, the k-major
    /// relabeling and the matrices each have to be doing real work.
    #[test]
    fn fused_ab8_negative_controls() {
        let t = table();
        let good = build_extend_mats_vslice(&t);
        let a = random_rows(0x2468_ACE0);
        let b = random_rows(0x1357_9BDF);
        let want: [[u8; ELL]; 8] = std::array::from_fn(|j| {
            window_oracle(&t, &a[j * 64..j * 64 + 64], &b[j * 64..j * 64 + 64])
        });

        let mut base = [[0u8; ELL]; 8];
        // SAFETY: both sources are 512 readable bytes.
        unsafe { fused_ab8_model(a.as_ptr(), b.as_ptr(), &good, &mut base) };
        assert_eq!(base, want, "unperturbed fused pipeline must match");

        // (a) perturbed matrices.
        for (idx, bit) in [(0usize, 5u32), (41, 17), (63, 62)] {
            let mut bad = good;
            bad[idx / 8][idx % 8] ^= 1u64 << bit;
            let mut got = [[0u8; ELL]; 8];
            // SAFETY: as above.
            unsafe { fused_ab8_model(a.as_ptr(), b.as_ptr(), &bad, &mut got) };
            assert_ne!(got, want, "mats[{idx}] bit {bit} flip left output unchanged");
        }

        // (b) swapping two windows' inputs must move the outputs, i.e. the
        // eight windows are genuinely independent and correctly addressed.
        let mut a_swapped = a;
        for i in 0..64 {
            a_swapped.swap(i, 64 + i);
        }
        let mut got = [[0u8; ELL]; 8];
        // SAFETY: as above.
        unsafe { fused_ab8_model(a_swapped.as_ptr(), b.as_ptr(), &good, &mut got) };
        assert_ne!(got, want, "swapping windows 0 and 1 of a left output unchanged");

        // (c) the `x^k` vector must not be uniform: reversing the K-row order
        // within window 0 changes which row gets which power of x.
        let mut a_krev = a;
        for k in 0..4 {
            for byte in 0..CHUNKS {
                a_krev.swap(k * CHUNKS + byte, (7 - k) * CHUNKS + byte);
            }
        }
        let mut got = [[0u8; ELL]; 8];
        // SAFETY: as above.
        unsafe { fused_ab8_model(a_krev.as_ptr(), b.as_ptr(), &good, &mut got) };
        assert_ne!(got[0], want[0], "reversing K-row order left window 0 unchanged");
    }

    /// The production entry point's addressing: window `j` reads stage bytes
    /// `64j..64j+64` and writes `out_base + j * out_stride`, and windows whose
    /// `live` bit is clear are left completely untouched. Same `octa_scatter`
    /// body the AVX-512 entry point runs.
    #[test]
    fn octa_scatter_addressing_and_live_mask() {
        let t = table();
        let mats = build_extend_mats_vslice(&t);
        let a = random_rows(0xC0DE_1234);
        let b = random_rows(0x5678_BEEF);
        // A stride wider than a window, as the real caller uses
        // (`BYTES_PER_BLOCK`), so a wrong stride cannot pass by aliasing.
        const STRIDE: usize = 192;
        for live in [0xffu32, 0x00, 0x01, 0x80, 0b1010_1010, 0b0101_0101, 0x0f] {
            let mut buf = vec![0xAAu8; 8 * STRIDE];
            // SAFETY: 512 readable bytes per stage; the buffer covers
            // `out_base + 7 * STRIDE + 64`.
            unsafe {
                round1_ab_inner_octa_model(
                    a.as_ptr(),
                    b.as_ptr(),
                    &mats,
                    buf.as_mut_ptr(),
                    STRIDE,
                    live,
                )
            };
            for j in 0..8 {
                let got = &buf[j * STRIDE..j * STRIDE + ELL];
                if live & (1 << j) == 0 {
                    assert!(got.iter().all(|&x| x == 0xAA), "live={live:#x} wrote dead {j}");
                } else {
                    let want =
                        window_oracle(&t, &a[j * 64..j * 64 + 64], &b[j * 64..j * 64 + 64]);
                    assert_eq!(got, &want[..], "live={live:#x} window={j}");
                }
                // Nothing may spill past the window into the stride padding.
                assert!(
                    buf[j * STRIDE + ELL..(j + 1) * STRIDE].iter().all(|&x| x == 0xAA),
                    "live={live:#x} window={j} wrote past 64 bytes"
                );
            }
        }
    }

    /// The 21-op fold butterfly must equal the 28-op transpose-then-XOR form
    /// it replaces, for arbitrary inputs — this is the only step of the fused
    /// pipeline whose shuffle network is new rather than reused.
    #[test]
    fn fold_butterfly_matches_transpose_then_xor() {
        let mut st = 0xF01D_u64;
        for _ in 0..64 {
            let s: [Bytes64; 8] =
                std::array::from_fn(|_| Bytes64(std::array::from_fn(|_| rng(&mut st) as u8)));
            let got = fold_final(fold_join(
                fold_pair(s[0], s[1]),
                fold_pair(s[2], s[3]),
            ), fold_join(fold_pair(s[4], s[5]), fold_pair(s[6], s[7])));
            assert_eq!(got, fold_k_reference(s));
            // And both really are "XOR the eight qwords of each plane".
            for u in 0..8 {
                let want = (0..8).fold(0u64, |acc, m| acc ^ s[u].qword(m));
                assert_eq!(got.qword(u), want, "u={u}");
            }
        }
    }

    /// `vgf2p8mulb`'s polynomial is the field's polynomial — the claim that no
    /// basis change is needed for the bilinear step. Checked exhaustively over
    /// all 65 536 byte pairs.
    #[test]
    fn mulb_model_is_the_field_product() {
        for x in 0..=255u8 {
            let xv = Bytes64([x; 64]);
            for y in 0..=255u8 {
                let got = xv.mulb(Bytes64([y; 64])).0[0];
                assert_eq!(got, (F8(x) * F8(y)).0, "x={x} y={y}");
            }
        }
    }

    /// The portable model of each instruction is a model of a *specific*
    /// instruction: pin the two whose encodings the kernel depends on.
    #[test]
    fn instruction_model_semantics() {
        // vpermb with BT is the 8x8 byte transpose and is an involution.
        let v = Bytes64(std::array::from_fn(|i| i as u8));
        let tr = v.permb(&BT);
        for j in 0..8 {
            for q in 0..8 {
                assert_eq!(tr.0[8 * j + q], (8 * q + j) as u8);
            }
        }
        assert_eq!(tr.permb(&BT), v);

        // vgf2p8affineqb with the identity matrix (row i at byte 7-i has bit i
        // set) is the identity map on bytes.
        let ident = u64::from_le_bytes([1, 2, 4, 8, 16, 32, 64, 128]).swap_bytes();
        assert_eq!(v.affine(ident), v);

        // qword_transpose really transposes the (vector, qword) index pair.
        let t: [Bytes64; 8] =
            std::array::from_fn(|i| Bytes64::from_qwords(std::array::from_fn(|j| (8 * i + j) as u64)));
        let p = qword_transpose(t);
        for j in 0..8 {
            for i in 0..8 {
                assert_eq!(p[j].qword(i), (8 * i + j) as u64);
            }
        }
    }

    /// The 512-byte matrix block is the whole working set: it must be derivable
    /// from the eight one-hot table rows alone.
    #[test]
    fn mats_depend_only_on_the_unit_columns() {
        let t = table();
        let mats = build_extend_mats(&t);
        // SAFETY: the table holds 256 rows of 64 readable bytes.
        let data = unsafe { core::slice::from_raw_parts(t.data_ptr(), 256 * ELL) };
        for (i_prime, m) in mats.iter().enumerate() {
            for bit in 0..8usize {
                let mut probe = Bytes64([1u8 << bit; 64]);
                probe = probe.affine(*m);
                assert_eq!(probe.0[0], data[(1usize << bit) * ELL + i_prime]);
            }
        }
    }
}
