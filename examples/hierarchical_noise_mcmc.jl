# examples/hierarchical_noise_mcmc.jl
#
# Trans-dimensional MCMC of a synthetic amplitude-only SMR sounding with the
# hierarchical heteroscedastic noise model of HierarchicalNoise.jl:
#
#     σᵢ² = s · σ̂ᵢ²
#
# The per-gate SHAPE σ̂ᵢ is handed to the sampler; the SCALE s is sampled
# jointly with the water-content model as a HiQGA nuisance parameter
# (log s, uniform within bounds) with a log-normal prior added in get_misfit.
# This is the hierarchical-Bayes device of Malinverno & Briggs (2004) and
# Malinverno & Parker (2006), as used inside trans-D samplers by Dettmer et
# al. (2010) and Bodin et al. (2012).
#
# The demonstration deliberately lies to the sampler: the data carry 50 nV
# noise but the asserted shape is 25 nV, so the truth is s = (50/25)² = 4.
# A working hierarchical model recovers s ≈ 4 from the data regardless of a
# prior centred at s = 1. The run is small (4 chains × 60 001 samples) so it
# finishes in a few minutes on a laptop; for a paper figure use more samples
# and the prior-centre sweep described in the module header (gate V3).
#
#     julia --project=. examples/hierarchical_noise_mcmc.jl

using Distributed
const NCHAINS = 4
# nworkers() is 1 with no workers (the master counts), so count real workers
nprocs() - 1 < NCHAINS && addprocs(NCHAINS - (nprocs() - 1))
@everywhere using Distributed, HiQGA.transD_GP, SMRPInversion

using SNMRForward, Statistics, Printf, PyPlot

# ── Synthetic: aquifer at 20–30 m, 25 % WC ─────────────────────────────────
zall, znall, zbnd = transD_GP.setupz(1.0, 1.028; dz=1.0, n=40, showplot=false)
w_true = fill(0.02, length(zbnd))
w_true[(zbnd .>= 20.0) .& (zbnd .< 30.0)] .= 0.25
qgrid  = 10.0 .^ range(log10(0.1), log10(15.0), length=24) |> collect

true_noise_nV   = 50.0
asserted_nV     = 25.0                      # wrong on purpose: s_true = 4
s_true          = (true_noise_nV / asserted_nV)^2

syn = SMRPInversion.make_synthetic_hierarchical(
    w_true, [0.05, 0.05], [500.0], 50_000e-9, deg2rad(-60.0), 50.0, zbnd, qgrid;
    noise_additive_nV = true_noise_nV,
    σ̂_V0_nV = fill(asserted_nV, length(qgrid)),
    s_prior_center = 1.0, s_prior_logwidth = 1.5,   # broad prior, centred at the wrong place
    s_init = 1.0, rseed = 7)
S = syn.S
@printf("Synthetic: peak V0 = %.0f nV, true noise %.0f nV, asserted shape %.0f nV → s_true = %.1f\n",
        1e9 * maximum(S.V0), true_noise_nV, asserted_nV, s_true)

# ── trans-D GP options (water content linear in [0.001, 0.5]) ───────────────
n_z       = length(zbnd)
fbounds   = [0.001 0.5]
xbounds   = [1.0 Float64(n_z)]
xall      = permutedims(collect(1:1.0:n_z))
opt = transD_GP.OptionsStat(
    nmin = 2, nmax = 30,
    xbounds = xbounds, fbounds = fbounds,
    fdataname = "hier_noise_example_",
    xall = xall, λ = [4.0], δ = 0.01,
    sdev_prop = 0.07 * diff(fbounds, dims=2)[:],
    sdev_pos  = [0.05 * abs(diff([extrema(znall)...])[1])],
    sdev_dc   = 0.008 * diff(fbounds, dims=2)[:],
    save_freq = 50, quasimultid = false, K = transD_GP.GP.OrstUhn())

# nuisance = [log s]; wide uniform bounds so the log-normal prior does the shaping
logs_bounds = [log(0.01) log(100.0)]
optn = transD_GP.OptionsNuisance(opt;
    sdev = [0.05], bounds = logs_bounds, updatenuisances = true)

# ── Sample ──────────────────────────────────────────────────────────────────
transD_GP.main(opt, optn, S; Tmax = 2.5, nsamples = 60_001,
               nchains = NCHAINS, nchainsatone = 1)
rmprocs(workers())

# ── Posterior of s and of w ────────────────────────────────────────────────
Mnu = transD_GP.assemblenuisancesatT(optn; burninfrac = 0.5, temperaturenum = 1)
s_post = exp.(vec(Mnu[:, 1]))
@printf("s posterior: median %.2f, p5–p95 %.2f–%.2f (truth %.1f, prior centre 1.0)\n",
        median(s_post), quantile(s_post, 0.05), quantile(s_post, 0.95), s_true)
# analytic reference when w is fixed at truth: s ≈ Σ r²/σ̂² / N
Vres, _ = SMRPInversion.applynu(S, w_true)
s_ref = sum((S.V0 .- Vres) .^ 2 ./ S.σ̂²_V0) / length(S.V0)
@printf("analytic s at w = truth: %.2f\n", s_ref)

opt.xall[:] .= zbnd
F = transD_GP.assembleTat1(opt, :fstar; temperaturenum = 1, burninfrac = 0.5)
samples = hcat([f[:] for f in F]...)
q05 = mapslices(x -> quantile(x, 0.05), samples; dims = 2)[:]
q50 = mapslices(x -> quantile(x, 0.50), samples; dims = 2)[:]
q95 = mapslices(x -> quantile(x, 0.95), samples; dims = 2)[:]

fig, ax = plt.subplots(1, 2, figsize=(9, 5.5), gridspec_kw=Dict("width_ratios" => [1.3, 1]))
ax[1].fill_betweenx(zbnd, q05, q95, alpha=0.3, color="steelblue", label="p5–p95")
ax[1].step(q50, zbnd, color="steelblue", lw=1.5, label="median")
ax[1].step(w_true, zbnd, "k-", lw=1.4, label="True")
ax[1].set_ylim(100, 0); ax[1].set_xlim(0, 0.4)
ax[1].set_xlabel("Water content"); ax[1].set_ylabel("Depth (m)")
ax[1].grid(true, alpha=0.3); ax[1].legend(fontsize=8, loc="lower right")
ax[1].set_title("trans-D posterior, hierarchical noise", fontsize=10)
ax[2].hist(s_post, bins=40, density=true, color="grey", alpha=0.7)
ax[2].axvline(s_true, color="k", lw=1.5, label="s truth")
ax[2].axvline(s_ref, color="crimson", ls="--", lw=1.2, label="analytic at w=truth")
ax[2].axvline(1.0, color="steelblue", ls=":", lw=1.2, label="prior centre")
ax[2].set_xlabel("noise scale s"); ax[2].legend(fontsize=8)
ax[2].set_title("posterior of s", fontsize=10)
fig.tight_layout()
savefig("examples/hierarchical_noise_mcmc.png", dpi=140, bbox_inches="tight")
plt.close("all")
@info "wrote examples/hierarchical_noise_mcmc.png"
