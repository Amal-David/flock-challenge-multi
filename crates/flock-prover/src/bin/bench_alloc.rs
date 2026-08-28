use std::time::Instant;
use rayon::prelude::*;

#[derive(Clone, Copy)]
pub struct F128 {
    pub lo: u64,
    pub hi: u64,
}
impl F128 {
    pub const ZERO: F128 = F128 { lo: 0, hi: 0 };
}

fn alloc_uninit(n: usize) -> Vec<F128> {
    let mut v = Vec::with_capacity(n);
    unsafe { v.set_len(n); }
    v
}

fn alloc_zero(n: usize) -> Vec<F128> {
    vec![F128::ZERO; n]
}

#[inline(never)]
fn do_work(v: &mut [F128]) {
    let stride = 64;
    for i in 0..64 {
        for j in 0..128 {
            v[j * stride + i] = F128 { lo: i as u64, hi: j as u64 };
        }
    }
}

fn bench(n: usize, iters: usize) {
    let t0 = Instant::now();
    (0..iters).into_par_iter().for_each(|_| {
        let mut v = alloc_zero(n);
        do_work(&mut v);
        std::hint::black_box(v.as_ptr());
    });
    println!("zero: {:?}", t0.elapsed());

    let t0 = Instant::now();
    (0..iters).into_par_iter().for_each(|_| {
        let mut v = alloc_uninit(n);
        do_work(&mut v);
        std::hint::black_box(v.as_ptr());
    });
    println!("uninit: {:?}", t0.elapsed());
}

fn main() {
    bench(8192, 1000000); // 128KB, highly parallel
}
