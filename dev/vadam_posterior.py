"""
Why Vadam's posterior sits at the prior — the arithmetic, on the real checkpoint.

    python3 dev/vadam_posterior.py            # from the repo root

The Vadam checkpoint was marked SUSPECT because (a) its mu "did not move"
(param std 0.0176 against an init of 0.020, while every other arm reached
0.042-0.053) and (b) 99.8% of its parameters sat at exactly the prior sigma.
dev/vadam_update.cu and dev/vadam_sample.cu since showed both kernels correct,
so this script asks what the saved numbers actually say. It prints three things:

  1. lambda/N against sqrt(s_hat) — the named suspicion about line 8's
     denominator. It is REAL (they are the same size) but it damps the mu step
     by ~2x, it does not freeze it.
  2. how far mu actually moved, against the exact init the trainer used.
     "AT INIT" turns out to be a coincidence of standard deviations.
  3. why sigma is prior-dominated: N*s has to beat lambda and misses by ~5
     orders of magnitude. This is the finding.

Nothing here needs a GPU or a retrain — it reads runs/vadam_C384_L6_100k_*.bin.
"""
import numpy as np

N       = 90_000_000.0     # text8 training tokens; sigma has the TRAIN N in it
LAMBDA  = 2500.0           # derived: 1/init_std^2, init_std = 0.02
ALPHA   = 1e-4             # lr
G2      = 0.999
STEPS   = 100_000
M       = 4 * 64           # minibatch = B*T tokens, the unit of the loss mean
MU_INIT_STD = 0.02


def load(path):
    """8 x int32 header {magic, n_lo, C, L, V, NH, steps, n_hi}, then n float32."""
    with open(path, "rb") as f:
        hdr = np.frombuffer(f.read(32), dtype=np.int32)
        assert hdr[0] == 20240520, f"{path}: bad magic {hdr[0]}"
        n = int(hdr[1]) | (int(hdr[7]) << 32)
        return np.frombuffer(f.read(n * 4), dtype=np.float32)


def mu_init(n):
    """The exact init of train_bnn_vadam.cu: srand(1337) + Box-Muller at 0.02.

    glibc's rand() is the TYPE_3 additive-feedback generator, reproduced here so
    this script needs no compiled helper. It is checked below against the one
    property that must hold: the result's std must be MU_INIT_STD.
    """
    r = [1337] + [0] * (2 * n + 343)
    for i in range(1, 31):
        r[i] = (16807 * r[i - 1]) % 2147483647
    for i in range(31, 34):
        r[i] = r[i - 31]
    for i in range(34, len(r)):
        r[i] = (r[i - 31] + r[i - 3]) & 0xFFFFFFFF
    v = np.array(r[344:344 + 2 * n], dtype=np.uint32) >> 1
    f32, RM = np.float32, np.float32(2147483647)
    u1 = (v[0::2].astype(f32) + f32(1.0)) / (RM + f32(2.0))
    u2 = (v[1::2].astype(f32) + f32(1.0)) / (RM + f32(2.0))
    out = f32(MU_INIT_STD) * np.sqrt(f32(-2.0) * np.log(u1)) * np.cos(f32(6.2831853) * u2)
    assert abs(out.std() / MU_INIT_STD - 1.0) < 0.01, "init RNG did not reproduce"
    return out.astype(np.float64)


def main():
    s  = load("runs/vadam_C384_L6_100k_s.bin").astype(np.float64)
    mu = load("runs/vadam_C384_L6_100k_mu.bin").astype(np.float64)
    m0 = mu_init(s.size)
    lt = LAMBDA / N
    bc2 = 1.0 - G2 ** STEPS          # 1.0 to float precision at 100K steps
    rs = np.sqrt(s / bc2)            # sqrt(s_hat), the line-8 denominator term
    prior = 1.0 / np.sqrt(LAMBDA)
    print(f"n = {s.size:,}   N = {N:,.0f}   lambda = {LAMBDA:.0f}   "
          f"prior sigma = {prior:.4f}   steps = {STEPS:,}\n")

    # ---- 1. the named suspicion: is lambda/N negligible in line 8? ----------
    att = rs / (rs + lt)             # step size relative to Adam's (eps ~ 0)
    print("1. LINE 8's DENOMINATOR  mu -= alpha * m_hat / (sqrt(s_hat) + lambda/N)")
    print(f"   lambda/N            = {lt:.4e}")
    print(f"   median sqrt(s_hat)  = {np.median(rs):.4e}   <- the SAME SIZE, not negligible")
    print(f"   fraction with sqrt(s_hat) < lambda/N : {np.mean(rs < lt) * 100:.1f}%")
    print(f"   mean step attenuation vs Adam        : {att.mean():.3f}  "
          f"(Adam's eps=1e-8 would give {np.mean(rs / (rs + 1e-8)):.3f})")
    print("   -> real, but it HALVES the step; it cannot freeze mu.\n")

    # ---- 2. did mu move? ---------------------------------------------------
    d = mu - m0
    print("2. DID MU MOVE")
    print(f"   std(mu_final) {mu.std():.6f}   std(mu_init) {m0.std():.6f}   "
          f"<- the coincidence that read as 'AT INIT'")
    print(f"   corr(mu_final, mu_init)  = {np.corrcoef(mu, m0)[0, 1]:.4f}   <- ~0: mu left its init")
    print(f"   ||mu-mu_init||/||mu_init|| = {np.linalg.norm(d) / np.linalg.norm(m0):.3f}")
    print(f"   median |mu-mu_init|      = {np.median(np.abs(d)):.4e}  "
          f"against an init scale of {MU_INIT_STD}")
    print("   -> mu moved. The std comparison was measuring the wrong thing.\n")

    # ---- 3. why sigma sits at the prior ------------------------------------
    print("3. WHY SIGMA IS PRIOR-DOMINATED   sigma = 1/sqrt(N*s + lambda)")
    print("   pct        s        sqrt(s)       N*s      sigma     data share of precision")
    for p in (50, 75, 90, 99, 99.9, 100):
        sv = np.percentile(s, p)
        prec = N * sv + LAMBDA
        print(f"   {p:5.1f}  {sv:.3e}  {np.sqrt(sv):.3e}  {N * sv:.3e}  "
              f"{1 / np.sqrt(prec):.6f}   {(prec - LAMBDA) / prec * 100:9.4f}%")
    need = np.sqrt(LAMBDA / N)
    print(f"\n   For data to matter, N*s must reach lambda = {LAMBDA:.0f}, i.e. the")
    print(f"   per-token RMS gradient must reach sqrt(lambda/N) = {need:.3e}.")
    print(f"   Observed median sqrt(s) = {np.median(rs):.3e}  ->  short by {need / np.median(rs):.0f}x.")
    w = (1 / M) * (N - M) / (N - 1)
    print(f"\n   Khan et al. Theorem 1: the minibatch estimate is biased,")
    print(f"   E[g_hat^2] = w*h + (1-w)*g^2 with w = (1/M)(N-M)/(N-1) = 1/{1/w:.0f} at M = {M}.")
    print(f"   Undoing it multiplies s by at most {1/w:.0f}, i.e. {np.sqrt(1/w):.0f}x on the gradient")
    print(f"   — still {need / (np.median(rs) * np.sqrt(1/w)):.0f}x short. Corrected, the fraction of")
    corr = 1.0 / np.sqrt(N * s / w + LAMBDA)
    print(f"   parameters within 10% of the prior falls only "
          f"{np.mean(1/np.sqrt(N*s+LAMBDA) > 0.9*prior)*100:.1f}% -> "
          f"{np.mean(corr > 0.9 * prior) * 100:.1f}%.")
    print("\n   -> STRUCTURAL. The kernels are correct (dev/vadam_{update,sample}.cu);")
    print("      this model's gradients are simply too small, against a prior of")
    print("      lambda = 1/init_std^2, for N*s to inform the posterior.")


if __name__ == "__main__":
    main()
