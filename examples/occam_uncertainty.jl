# examples/occam_uncertainty.jl
#
# Deterministic (Occam) inversion of a synthetic amplitude-only SMR sounding
# with the three uncertainty views the package offers, side by side:
#
#   1. the Occam model itself (Constable, Parker & Constable 1987) and
#      whether it reached χ² = N;
#   2. the linearised (Laplace) ±2σ band from the Gauss-Newton Hessian,
#      together with the model-resolution diagonal that says where that
#      band is data-driven (Tarantola 2005; Alumbaugh & Newman 2000);
#   3. a parametric bootstrap p5–p95 band (Efron & Tibshirani 1993).
#
# No MCMC, no workers; runs in a minute or two:
#
#     julia --project=. examples/occam_uncertainty.jl
#
# What to look for: the Laplace band balloons below the aquifer where
# diag(R) → 0 (prior-dominated), the bootstrap band stays tight there
# (noise-only, no smoothing bias), and the recovered ∫w dz differs from the
# truth — all known properties of regularised inversion, not bugs. On this
# noise realisation Occam also parks a smooth ramp of water below 70 m, where
# the data cannot object and the depth-scaled roughness barely charges for it;
# ∫w dz comes out three times too large. The script prints what the pure CPC87
# roughness (anchor_top = false) does (the same) and what a small Marquardt
# damping towards the reference model (β² = 0.01) does (removes it).

using SMRPInversion, SNMRForward, Statistics, Printf, PyPlot
using HiQGA: transD_GP

# ── Synthetic aquifer at 20–30 m, 25 % WC, resistive half-space ────────────
zall, znall, zbnd = transD_GP.setupz(1.0, 1.028; dz=1.0, n=60, showplot=false)
w_true = fill(0.02, length(zbnd))
w_true[(zbnd .>= 20.0) .& (zbnd .< 30.0)] .= 0.25

qgrid  = 10.0 .^ range(log10(0.1), log10(15.0), length=24) |> collect
L      = 50.0            # 100 m square loop → half-side
φ_inc  = deg2rad(-60.0)
Be     = 50_000e-9
σ_lay  = [0.05, 0.05]; t_lay = [500.0]
noise  = 50e-9

make(domain) = SMRPInversion.create_synthetic(
    w_true, σ_lay, t_lay, Be, φ_inc, L, zbnd, qgrid;
    noise_additive = noise, noise_domain = domain,
    mult = false, amponly = true, linearsat = true,
    noise_mle = false, showplot = false, rseed = 42)
sounding = make(:amplitude)          # Gaussian noise on |V|: what the likelihood assumes
sounding_c = make(:complex)          # Gaussian noise on each quadrature, then |·|: Rician

# trapezoidal weights to report ∫w dz
dz = sounding.Fm.dz
wt = [0.5dz[1]; 0.5 .* (dz[1:end-1] .+ dz[2:end]); 0.5dz[end]]
∫(w) = sum(wt .* w)

# ── 1. Occam + Laplace ──────────────────────────────────────────────────────
n_z = length(zbnd)
r = SMRPInversion.deterministic_invert(sounding;
        wstart = fill(0.02, n_z), wref = fill(0.02, n_z),
        regtype = :R1, uncertainty = :laplace)
@printf("Occam: χ² = %.2f (N = %d), converged = %s, log10 λ² = %.2f\n",
        r.χ²_final, length(sounding.V0), r.converged, log10(r.λ²_final))
@printf("       ∫w dz = %.2f m (truth %.2f m), peak w = %.3f (truth 0.25)\n",
        ∫(r.w), ∫(w_true), maximum(r.w))
σ_lap  = r.uncertainty.σ
diagR  = r.uncertainty.diagR
@printf("       mean diag(R): <30 m %.2f, >60 m %.2f\n",
        mean(diagR[zbnd .< 30]), mean(diagR[zbnd .> 60]))

# ── 1a. Pure CPC87 roughness: the deep null-space ramp ──────────────────────
ru = SMRPInversion.deterministic_invert(sounding;
        wstart = fill(0.02, n_z), wref = fill(0.02, n_z), regtype = :R1, anchor_top = false)
@printf("CPC87 R1 (anchor_top = false): χ² = %.2f, log10 λ² = %.2f, ∫w dz = %.2f m, mean w below 70 m = %.3f (truth 0.02)\n",
        ru.χ²_final, log10(ru.λ²_final), ∫(ru.w), mean(ru.w[zbnd .> 70]))

# ── 1a'. Reference-model damping removes the ramp ───────────────────────────
rb = SMRPInversion.deterministic_invert(sounding;
        wstart = fill(0.02, n_z), wref = fill(0.02, n_z), regtype = :R1, β² = 0.01)
@printf("β² = 0.01 towards wref: χ² = %.2f, converged = %s, ∫w dz = %.2f m, mean w below 70 m = %.3f\n",
        rb.χ²_final, rb.converged, ∫(rb.w), mean(rb.w[zbnd .> 70]))

# ── 1b. The same inversion on Rician (complex-domain) noise ─────────────────
# The Gaussian-in-|V| likelihood is the high-SNR limit. The low-q gates here
# have SNR of only a few, where the Rician amplitude is biased high (Rice
# 1944), and Occam fits that extra "signal" with extra water.
rc = SMRPInversion.deterministic_invert(sounding_c;
        wstart = fill(0.02, n_z), wref = fill(0.02, n_z), regtype = :R1)
@printf("Rician noise: min gate SNR %.1f; χ² = %.2f, converged = %s, ∫w dz = %.2f m (truth %.2f m)\n",
        minimum(sounding_c.V0 ./ sounding_c.σ_V0), rc.χ²_final, rc.converged, ∫(rc.w), ∫(w_true))
@printf("              (Gaussian-amplitude noise, same seed: ∫w dz = %.2f m)\n", ∫(r.w))

# ── 2. Bootstrap (same regularisation settings) ─────────────────────────────
b = SMRPInversion.bootstrap_invert(sounding; N = 60, w0 = r.w,
        method = :parametric, noise_domain = :amplitude, rseed = 1,
        wstart = fill(0.02, n_z), wref = fill(0.02, n_z), regtype = :R1)
@printf("Bootstrap: %d/%d ran, %d reached χ² target; log10 λ² spread sd = %.2f\n",
        b.n_ok, 60, b.n_converged, std(log10.(filter(!isnan, b.λ²_reps))))
∫reps = [∫(b.w_reps[:, i]) for i in 1:size(b.w_reps, 2) if !isnan(b.w_reps[1, i])]
@printf("           ∫w dz p5–p95 = %.2f–%.2f m (truth %.2f m)\n",
        quantile(∫reps, 0.05), quantile(∫reps, 0.95), ∫(w_true))

# ── Figure ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 3, figsize=(11, 6.5), sharey=true,
                       gridspec_kw=Dict("width_ratios" => [2, 2, 1]))
lo2σ = clamp.(r.w .- 2σ_lap, 0, 0.4); hi2σ = clamp.(r.w .+ 2σ_lap, 0, 0.4)
ax[1].fill_betweenx(zbnd, lo2σ, hi2σ, alpha=0.25, color="darkorange", label="Laplace ±2σ (clipped)")
ax[1].step(r.w, zbnd, color="darkorange", lw=1.8, label="Occam R1")
ax[1].step(rb.w, zbnd, color="green", lw=1.2, ls="--", label="Occam R1, β²=0.01")
ax[1].step(w_true, zbnd, "k-", lw=1.4, label="True")
ax[1].set_title("Occam + linearised band"); ax[1].legend(fontsize=8, loc="lower right")

ax[2].fill_betweenx(zbnd, b.q05, b.q95, alpha=0.3, color="steelblue", label="Bootstrap p5–p95")
ax[2].step(b.q50, zbnd, color="steelblue", lw=1.5, label="Bootstrap median")
ax[2].step(r.w, zbnd, color="darkorange", lw=1.2, ls="--", label="Occam R1")
ax[2].step(w_true, zbnd, "k-", lw=1.4, label="True")
ax[2].set_title("Bootstrap (N = 60)"); ax[2].legend(fontsize=8, loc="lower right")

ax[3].plot(diagR, zbnd, color="purple", lw=1.6)
ax[3].set_xlim(0, 1); ax[3].set_title("diag(R)"); ax[3].set_xlabel("resolution")
for a in ax[1:2]
    a.set_xlim(0, 0.35); a.set_xlabel("Water content"); a.grid(true, alpha=0.3)
end
ax[3].grid(true, alpha=0.3)
ax[1].set_ylim(120, 0); ax[1].set_ylabel("Depth (m)")
fig.suptitle("Occam uncertainty views: synthetic 10 m aquifer at 20 m, 50 nV noise", fontsize=10)
fig.tight_layout()
savefig("examples/occam_uncertainty.png", dpi=140, bbox_inches="tight")
plt.close("all")
@info "wrote examples/occam_uncertainty.png"
