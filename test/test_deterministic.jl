# Tests for SMRDetOperator + deterministic_invert
#
# Covers:
#   1. Analytical amplitude-Jacobian matches a finite-difference reference
#      at a representative model. Numerical validation of the derivation
#      J[j,i] = Re(conj(V_j) · K̃[j,i]) / |V_j|.
#   2. `deterministic_invert` converges on a synthetic aquifer, hitting
#      the χ² = n_data target within reasonable tolerance and recovering
#      the aquifer bump in the right depth range.

using Test, LinearAlgebra, SparseArrays, Statistics
using SMRPInversion, SNMRForward
using HiQGA: transD_GP

# ── Reusable synthetic setup ────────────────────────────────────────────────
"""
Build a fresh synthetic sounding with a 10 m aquifer at 20 m depth,
25 % water content, in a resistive half-space. Returns the sounding, the
true water content vector on the inversion grid, and the depth grid.
"""
function synthetic_sounding(; noise_V = 100e-9)
    # depth grid, log-growing to ~200 m
    zall, znall, zbnd = transD_GP.setupz(1.0, 1.028; dz=1.0, n=60, showplot=false)

    # true water content: 25 % between 20 and 30 m, 2 % elsewhere
    w_true = fill(0.02, length(zbnd))
    w_true[(zbnd .>= 20.0) .& (zbnd .< 30.0)] .= 0.25

    # resistive halfspace
    σ_layers = [0.05, 0.05]
    t_layers = [500.0]

    # 24 log-spaced pulse moments 0.1 – 15 A·s
    qgrid = 10.0 .^ range(log10(0.1), log10(15.0), length=24) |> collect

    # 100 m square loop (L is half-side per SNMRForward convention),
    # inclination −60°, Earth field 50 μT
    L, φ_inc, θ_loop, Be = 50.0, deg2rad(-60.0), 0.0, 50_000e-9

    S = SMRPInversion.create_synthetic(
        w_true, σ_layers, t_layers, Be, φ_inc, L, zbnd, qgrid;
        noise_additive = noise_V,
        mult = false, amponly = true, linearsat = true, noise_mle = false,
        showplot = false, rseed = 42,
    )
    return S, w_true, zbnd
end

# Instantiate once for reuse across sub-testsets
sounding, w_true, zbnd = synthetic_sounding()

# ────────────────────────────────────────────────────────────────────────────
@testset "Analytical Jacobian matches finite differences" begin
    op = SMRPInversion.SMRDetOperator(sounding)
    w_eval = copy(w_true)

    # Populate op.J with the analytical Jacobian at w_eval
    transD_GP.AbstractOperator.getresidual(op, w_eval; computeJ = true)
    J_ana = copy(op.J)

    # Central-difference finite Jacobian
    nq, nz = size(J_ana)
    J_fd = zeros(nq, nz)
    h    = 1e-6
    for i in 1:nz
        wp = copy(w_eval); wp[i] += h
        wm = copy(w_eval); wm[i] -= h
        transD_GP.AbstractOperator.getresidual(op, wp; computeJ = false)
        ap = copy(op.res) .+ sounding.V0    # forward |V| = res + V0
        transD_GP.AbstractOperator.getresidual(op, wm; computeJ = false)
        am = copy(op.res) .+ sounding.V0
        J_fd[:, i] .= (ap .- am) ./ (2h)
    end
    # Restore op.J
    transD_GP.AbstractOperator.getresidual(op, w_eval; computeJ = true)

    # Comparison: Frobenius relative error should be < 1e-6 (limited by float64
    # + central-difference step size, not by the analytical formula)
    frob_rel = norm(J_ana .- J_fd) / norm(J_ana)
    @test frob_rel < 1e-6
end

# ────────────────────────────────────────────────────────────────────────────
@testset "depth-scaled makereg for SMRDetOperator" begin
    op = SMRPInversion.SMRDetOperator(sounding)
    dz = op.S.Fm.dz
    nz = size(op.K̃, 2)

    R0 = transD_GP.makereg(:R0, op)
    @test size(R0) == (nz, nz)
    @test R0 == sparse(1.0I, nz, nz)

    # Default: top cell anchored (a weak minimum-norm pull through the chain
    # of differences). Row 1 has a single non-zero on the diagonal.
    R1 = transD_GP.makereg(:R1, op)
    @test size(R1) == (nz, nz)
    @test R1[1, 1] ≈ 1.0 / sqrt(dz[1])
    @test count(!iszero, R1[1, :]) == 1

    # For i ≥ 2, row i is the scaled first-difference with weight 1/√dz[i-1],
    # so ‖R1 w‖² ≈ ∫(∂w/∂z)² dz: check on a linear profile w = a z (plus the
    # anchor term a² z₁²/dz₁ for the default form)
    for i in 2:nz
        @test R1[i, i-1] ≈ -1.0 / sqrt(dz[i-1])
        @test R1[i, i]   ≈  1.0 / sqrt(dz[i-1])
    end
    a = 0.003
    wlin = a .* op.S.Fm.zgrid
    @test norm(R1 * wlin)^2 ≈ a^2 * sum(dz) + (a * op.S.Fm.zgrid[1])^2 / dz[1] rtol = 1e-10

    # Deep rows have smaller magnitude than shallow rows (dz grows with
    # depth on a log-growing grid), so the deep smoothness is looser.
    @test abs(R1[2, 2]) > abs(R1[nz, nz])

    # Pure CPC87 form: row 1 zero, constant models in the null space
    opu = SMRPInversion.SMRDetOperator(sounding; anchor_top = false)
    R1u = transD_GP.makereg(:R1, opu)
    @test count(!iszero, R1u[1, :]) == 0
    @test norm(R1u * ones(nz)) ≈ 0 atol = 1e-12
    @test norm(R1u * wlin)^2 ≈ a^2 * sum(dz) rtol = 1e-10

    # R2 annihilates linear profiles and is nz × nz
    R2 = transD_GP.makereg(:R2, op)
    wlin = a .* op.S.Fm.zgrid
    @test size(R2) == (nz, nz)
    @test norm(R2 * wlin) ≈ 0 atol = 1e-10
    @test norm(R2 * ones(nz)) ≈ 0 atol = 1e-12
    @test norm(R2 * (op.S.Fm.zgrid .^ 2)) > 0
end

# ────────────────────────────────────────────────────────────────────────────
@testset "create_synthetic loop flag and half-side convention" begin
    zall, znall, zb = transD_GP.setupz(1.0, 1.028; dz=1.0, n=60, showplot=false)
    q  = 10.0 .^ range(log10(0.1), log10(15.0), length=12) |> collect
    ct = SNMRForward.ConductivityModel([0.05, 0.05], [500.0])
    Be, inc = 50_000e-9, deg2rad(-60.0)
    wshal = zeros(length(zb)); wshal[zb .< 10.0] .= 0.2

    S_sq = SMRPInversion.create_synthetic(wshal, [0.05,0.05], [500.0], Be, inc, 50.0, zb, q;
                                           square = true, showplot = false, amponly = true, linearsat = true)
    S_ci = SMRPInversion.create_synthetic(wshal, [0.05,0.05], [500.0], Be, inc, 50.0, zb, q;
                                           square = false, showplot = false, amponly = true, linearsat = true)
    Fsq = SNMRForward.MRSForward_square(50.0, zb, q, inc, 0.0, Be, ct)
    Fci = SNMRForward.MRSForward(50.0, zb, q, inc, Be, ct)
    @test S_sq.Fm.kernel ≈ Fsq.kernel
    @test S_ci.Fm.kernel ≈ Fci.kernel
    @test !(S_sq.Fm.kernel ≈ Fci.kernel)

    # L is the HALF-side: a square loop with L = 50 has the area of a circle
    # with R = 2L/√π and the two peak responses agree to ~15 % (shape still
    # matters for a shallow target), while a circle of area L² (R = L/√π),
    # which is what L = full side would imply, is off by a factor > 2.
    V_sq  = abs.(SNMRForward.forward(Fsq, wshal))
    V_eqA = abs.(SNMRForward.forward(SNMRForward.MRSForward(2 * 50.0 / sqrt(pi), zb, q, inc, Be, ct), wshal))
    V_sm  = abs.(SNMRForward.forward(SNMRForward.MRSForward(50.0 / sqrt(pi), zb, q, inc, Be, ct), wshal))
    @test isapprox(maximum(V_sq), maximum(V_eqA); rtol = 0.25)
    @test maximum(V_sq) / maximum(V_sm) > 2.0

    # complex-domain noise is Rician: mean amplitude biased high at low SNR
    S_c = SMRPInversion.create_synthetic(wshal, [0.05,0.05], [500.0], Be, inc, 50.0, zb, q;
                                          noise_additive = 1e-3, noise_domain = :complex,
                                          showplot = false, amponly = true, linearsat = true)
    @test all(S_c.V0 .>= 0)
    @test mean(S_c.V0) > 0.5e-3
end

# ────────────────────────────────────────────────────────────────────────────
@testset "deterministic_invert converges to χ² target" begin
    n = length(zbnd)
    wref   = fill(0.02, n)
    wstart = fill(0.02, n)

    r = SMRPInversion.deterministic_invert(
        sounding; wstart = wstart, wref = wref, regtype = :R1)
    op, mnew, χ², λ², oidx = r

    # final iterate
    wfinal    = mnew[end][oidx[end]]
    χ²_final  = χ²[end][oidx[end]]
    target    = length(sounding.V0)
    @test r.w == wfinal
    @test r.χ²_final == χ²_final
    @test r.λ²_final == λ²[end][oidx[end]][1]

    # χ² should be within ~10% of target for a well-posed problem
    @test isapprox(χ²_final, target; rtol = 0.1)
    @test r.converged == (χ²_final <= target * (1 + 1e-2))

    # op.J and op.res are left at the final model
    @test op.ρ == wfinal
    @test op.res ≈ abs.(op.K̃ * wfinal) .- sounding.V0

    # Solution should live in the physical bounds
    @test all(1e-4 .<= wfinal .<= 0.5)

    # Iterations shouldn't blow up
    @test length(mnew) <= 15
end

# ────────────────────────────────────────────────────────────────────────────
@testset "deterministic_invert recovers aquifer amplitude" begin
    n = length(zbnd)
    wref   = fill(0.02, n)
    wstart = fill(0.02, n)

    op, mnew, χ², λ², oidx = SMRPInversion.deterministic_invert(
        sounding; wstart = wstart, wref = wref, regtype = :R1)
    wfinal = mnew[end][oidx[end]]

    # Aquifer sits at 20–30 m in the truth. Occam's smoothness broadens it,
    # so the recovered peak in a window around the aquifer should be well
    # above the 2 % background but need not match the true 25 % exactly.
    aquifer_mask = (zbnd .>= 15.0) .& (zbnd .<= 40.0)
    peak_in_aquifer = maximum(wfinal[aquifer_mask])
    @test peak_in_aquifer > 0.10
    @test peak_in_aquifer < 0.35

    # Away from the aquifer (below 60 m), we should stay well below the
    # aquifer amplitude. The depth-scaled R matrix penalises deep
    # differences less than shallow (deep dz is larger), so the aquifer
    # tail spreads noticeably into the deep — allow up to ~2/3 of the
    # aquifer peak before flagging as runaway.
    deep_mask = zbnd .> 60.0
    @test maximum(wfinal[deep_mask]) < 0.20
end

# ────────────────────────────────────────────────────────────────────────────
@testset "linearised_posterior_covariance" begin
    n = length(zbnd)
    wref   = fill(0.02, n)
    wstart = fill(0.02, n)
    op, mnew, χ², λ², oidx = SMRPInversion.deterministic_invert(
        sounding; wstart = wstart, wref = wref, regtype = :R1)

    λ²_final = λ²[end][oidx[end]][1]

    # Compute at the same regtype used in inversion
    result = SMRPInversion.linearised_posterior_covariance(op, λ²_final;
                                                            regtype = :R1)

    # Shape sanity
    @test size(result.Cm) == (n, n)
    @test length(result.σ) == n
    @test result.λ² ≈ float(λ²_final)

    # Resolution matrix: C_m JᵀW²J is not symmetric, so its diagonal can dip
    # slightly below zero where the regulariser dominates; it should be O(1)
    # at best and better resolved shallow than deep
    @test size(result.R) == (n, n)
    @test all(result.diagR .>= -0.1)
    @test maximum(result.diagR) <= 1.5
    @test mean(result.diagR[zbnd .< 30]) > mean(result.diagR[zbnd .> 60])

    # β² enters the Hessian: a positive β² tightens the covariance
    result_β = SMRPInversion.linearised_posterior_covariance(op, λ²_final; regtype = :R1, β² = 1.0)
    @test mean(result_β.σ) < mean(result.σ)

    # Uncertainty is non-negative and finite everywhere
    @test all(isfinite, result.σ)
    @test all(result.σ .>= 0.0)

    # Kernel sensitivity dies with depth, so σ should be larger deep than
    # shallow (SMR resolvability limit). Use the top/bottom 25 % of the
    # grid so this holds regardless of grid extent.
    n_split = length(zbnd) ÷ 4
    shallow_σ = maximum(result.σ[1:n_split])
    deep_σ    = maximum(result.σ[end - n_split + 1:end])
    @test deep_σ > shallow_σ

    # Increasing λ² tightens the posterior (more smoothing → smaller σ per
    # cell where the smoothness regulariser is active)
    result_looser  = SMRPInversion.linearised_posterior_covariance(
        op, λ²_final / 10; regtype = :R1)
    result_tighter = SMRPInversion.linearised_posterior_covariance(
        op, λ²_final * 10; regtype = :R1)
    @test mean(result_tighter.σ) < mean(result_looser.σ)
end
