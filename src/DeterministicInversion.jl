# Deterministic (Occam-style Gauss-Newton) SMR inversion.
#
# Wraps an `SMRSoundingKnown` into a HiQGA `Operator1D` that
# `HiQGA.transD_GP.gradientinv` can consume. The SMR forward is
#     V(q) = K̃(q, z) · w(z)
# — linear in water content `w`, complex-valued in general. This module
# implements the amplitude-only inversion, |V(q)| = |K̃·w|, using the
# analytical amplitude-Jacobian
#
#     J[j, i] = ∂|V_j| / ∂w_i = Re(conj(V_j) · K̃[j, i]) / |V_j|,
#
# which is the chain rule applied to |V| = sqrt(V · conj(V)) with V linear in w.
#
# References
# ----------
# * Occam / smoothest-model inversion with a χ² = N target:
#   Constable, Parker & Constable (1987), Geophysics 52, 289–300,
#   doi:10.1190/1.1442303. HiQGA's `gradientinv` provides the outer line
#   search over λ² and the Gauss-Newton mechanics; the damped step towards a
#   reference model (β² > 0) is Marquardt (1963), SIAM J. Appl. Math. 11, 431.
# * Roughness discretisation on a non-uniform grid (1/√Δz row weights so
#   ‖R w‖² ≈ ∫(∂w/∂z)² dz): Constable et al. (1987) eq. 3; Aster, Borchers &
#   Thurber (2018), Parameter Estimation and Inverse Problems, 3rd ed., Ch. 4.
# * Why amplitude-only SMR inversion is non-linear while complex-data
#   inversion is linear: Weichman, Lavely & Ritzwoller (2000), Phys. Rev. E
#   62, 1290; Braun, Hertrich & Yaramanci (2005), Near Surf. Geophys. 3, 155;
#   Müller-Petke & Yaramanci (2010), Geophysics 75, WA199.
# * Linearised posterior covariance and model resolution:
#   Tarantola (2005), Inverse Problem Theory, SIAM, eqs 3.37–3.40;
#   Menke (2018), Geophysical Data Analysis, Ch. 5;
#   Alumbaugh & Newman (2000), Geophysics 65, 1455–1467.
#
# Noise-model caveat: the likelihood is Gaussian in |V|. For a complex signal
# with Gaussian quadrature noise the amplitude is Rician (Rice 1944), which is
# biased high by ≈ +2 % at SNR 5, +14 % at SNR 2 and +55 % at SNR 1. Use the
# complex data (`amponly = false`) where low-SNR gates matter.

using LinearAlgebra, SparseArrays

export SMRDetOperator, deterministic_invert, linearised_posterior_covariance

"""
    SMRDetOperator <: HiQGA.transD_GP.AbstractOperator.Operator1D

Deterministic-inversion wrapper around an `SMRSoundingKnown`, satisfying the
interface HiQGA's Occam-style `gradientinv` expects.

Fields:
- `S`      : the underlying SMR sounding (data, noise, forward object).
- `K̃`      : `(nq × nz)` complex kernel matrix such that `V = K̃ * w`
             (SNMRForward kernel × trapezoidal quadrature weights on `zgrid`).
             `w` is a nodal field on `S.Fm.zgrid`, integrated with the
             trapezoidal rule exactly as `SNMRForward.forward` does.
- `ρ`      : latest evaluated model iterate (kept in the field HiQGA expects).
- `nfixed` : number of fixed depth cells at the top (always 0 here).
- `J`      : `(nq × nz)` real amplitude Jacobian, populated in-place by
             `getresidual(op, w; computeJ=true)`.
- `W`      : `diag(1 ./ σ_V0)`, sparse — used for chi-square weighting.
- `res`    : `(nq,)` current residual `|V(w)| − V0`, updated by
             `getresidual`.
- `anchor_top` : if `true` (default), row 1 of the `:R1` roughness matrix
             penalises the top cell towards zero, which through the chain of
             differences acts as a weak minimum-norm pull on the whole model
             (see `makereg`). `false` gives the pure CPC87 form.
"""
mutable struct SMRDetOperator <: transD_GP.AbstractOperator.Operator1D
    S          :: SMRSoundingKnown
    K̃          :: Matrix{ComplexF64}
    ρ          :: Vector{Float64}
    nfixed     :: Int
    J          :: Matrix{Float64}
    W          :: SparseMatrixCSC{Float64,Int}
    res        :: Vector{Float64}
    anchor_top :: Bool
end

"""
    SMRDetOperator(S::SMRSoundingKnown; anchor_top = true) -> SMRDetOperator

Build a deterministic-inversion operator from an amplitude-only SMR sounding.

Precomputes `K̃ = trapezoidal_weights .* kernel'` so the amplitude forward is
`|K̃ * w|`. Throws if `S.amponly` is false — the analytical Jacobian is
amplitude-specific.
"""
function SMRDetOperator(S::SMRSoundingKnown; anchor_top::Bool = true)
    S.amponly || throw(ArgumentError("SMRDetOperator currently supports amplitude-only inversion."))
    kernel = S.Fm.kernel                       # (nz × nq) complex
    dz     = S.Fm.dz                           # (nz-1,) layer thicknesses
    nz, nq = size(kernel)
    nz == length(S.Fm.zgrid) || error("kernel row count must match zgrid length")

    # Trapezoidal quadrature weights at each depth node (same rule as SNMRForward.forward)
    wt = zeros(Float64, nz)
    wt[1]   = 0.5 * dz[1]
    wt[end] = 0.5 * dz[end]
    @inbounds for i in 2:nz-1
        wt[i] = 0.5 * (dz[i-1] + dz[i])
    end
    K̃ = permutedims(wt .* kernel)              # (nq × nz), so V = K̃ * w

    J   = zeros(Float64, nq, nz)
    W   = sparse(Diagonal(1.0 ./ S.σ_V0))
    res = zeros(Float64, nq)
    ρ   = zeros(Float64, nz)
    SMRDetOperator(S, K̃, ρ, 0, J, W, res, anchor_top)
end

"""
    getresidual(op::SMRDetOperator, w::Vector{Float64}; computeJ=false)

Evaluate the amplitude-only forward at model `w`, populate `op.res` with the
signed residual `|K̃·w| − V0`, and — if `computeJ=true` — populate `op.J` with
the analytical amplitude Jacobian
`J[j,i] = Re(conj(V[j]) · K̃[j,i]) / |V[j]|`.

Implements `HiQGA.transD_GP.AbstractOperator.getresidual`.
"""
function transD_GP.AbstractOperator.getresidual(op::SMRDetOperator, w::Vector{Float64}; computeJ=false)
    V   = op.K̃ * w
    amp = abs.(V)
    @inbounds for j in eachindex(amp)
        op.res[j] = amp[j] - op.S.V0[j]
    end
    if computeJ
        # Floor on |V_j| prevents NaN when the complex kernel row nearly
        # cancels for a particular w. In that regime the amplitude
        # derivative is genuinely very large (small perturbations to w
        # produce a large relative change in |V|), so a large-but-finite
        # J is the right physical answer.
        @inbounds for i in axes(op.K̃, 2), j in axes(op.K̃, 1)
            denom = max(amp[j], 1e-30)
            op.J[j,i] = real(conj(V[j]) * op.K̃[j,i]) / denom
        end
    end
    op.ρ .= w
    nothing
end

"""
    makereg(r::Symbol, F::SMRDetOperator) -> SparseMatrixCSC

Depth-scaled roughness operators for `SMRDetOperator`, dispatched over
HiQGA's generic `makereg`.

- `:R0` — identity (smallest model).
- `:R1` — first difference with row weights `1/√dz[i-1]`, so that
  `‖R₁w‖² ≈ ∫(∂w/∂z)² dz` on the log-growing grid from
  `transD_GP.setupz` (Constable, Parker & Constable 1987 eq. 3; Aster et
  al. 2018 Ch. 4).
  With `F.anchor_top = true` (default, the behaviour of the original
  implementation) row 1 is `w[1]/√dz[1]`, which through the chain of
  differences is a weak pull of the whole model towards zero at large
  λ². With `F.anchor_top = false` row 1 is all zeros and constant models
  lie in the null space: the pure CPC87 form. In practice the two give
  almost the same model; only the top cell differs.

  **Deep null-space ramp.** Neither form stops the inversion from parking
  a smooth ramp of water below the depth of resolution when a noise
  realisation asks for a little extra high-q signal: deep water is cheap
  in roughness (the 1/√dz weights get small) and the data cannot object
  (SMR sensitivity falls off with depth; Hertrich 2008). On the test
  synthetic (10 m aquifer at 20 m, seed 42) this triples ∫w dz while
  χ² = N is met. The remedy is reference-model damping: `β² = 0.01`
  towards `wref` (Marquardt 1963) removes the ramp on that case with
  χ² = N still reached. Always report `diagR` and ∫w dz with an Occam
  model.
- `:R2` — second difference `D₂ = diag(1/√dz_c) · diff(diag(1/dz) · diff(·))`
  with `dz_c` the node spacing, so that `‖R₂w‖² ≈ ∫(∂²w/∂z²)² dz`.
"""
function transD_GP.makereg(r::Symbol, F::SMRDetOperator)
    dz = F.S.Fm.dz
    nz = size(F.K̃, 2)
    length(dz) == nz - 1 || error("dz length ($(length(dz))) must be nz-1 ($(nz-1))")
    r === :R0 && return sparse(1.0I, nz, nz)
    if r === :R1
        rows = Int[]; cols = Int[]; vals = Float64[]
        if F.anchor_top
            push!(rows, 1); push!(cols, 1); push!(vals, 1.0 / sqrt(dz[1]))
        end
        @inbounds for i in 2:nz
            s = 1.0 / sqrt(dz[i-1])
            push!(rows, i); push!(cols, i-1); push!(vals, -s)
            push!(rows, i); push!(cols, i);   push!(vals,  s)
        end
        return sparse(rows, cols, vals, nz, nz)
    end
    if r === :R2
        # first derivative at the nz-1 cell centres, then its derivative at the
        # nz-2 interior nodes, weighted by sqrt of the node spacing
        D1 = spdiagm(nz-1, nz, 0 => -1.0 ./ dz, 1 => 1.0 ./ dz)
        dzc = 0.5 .* (dz[1:end-1] .+ dz[2:end])            # spacing between cell centres
        D2 = spdiagm(nz-2, nz-1, 0 => -1.0 ./ dzc, 1 => 1.0 ./ dzc)
        Wc = spdiagm(sqrt.(dzc))
        R2 = Wc * D2 * D1                                      # (nz-2) × nz
        # pad to nz × nz with two zero rows so HiQGA's R'R is nz × nz
        return vcat(spzeros(2, nz), R2)
    end
    error("unknown regularization: $r")
end

"""
    deterministic_invert(S::SMRSoundingKnown; wstart, wref, uncertainty = :none, kwargs...)
        -> NamedTuple (op, mnew, χ², λ², oidx, uncertainty, w, χ²_final, λ²_final, converged)

Run Occam-style Gauss-Newton inversion on an amplitude-only SMR sounding
(Constable, Parker & Constable 1987): find the smoothest model that fits the
data to within `target` (default χ² = N).

Uses the analytical Jacobian above; delegates the outer line-search over the
smoothness weight `λ²` to `HiQGA.transD_GP.gradientinv`.

Arguments:
- `S`      : sounding with known noise (`SMRSoundingKnown`).
- `wstart` : starting water-content vector, one entry per node in
             `S.Fm.zgrid`. Default `0.05` everywhere.
- `wref`   : reference model used only when `β² > 0`. Default `0.05`.
- `lo`, `hi` : per-cell bounds on water content. Default `[1e-4, 0.4]`.
              `hi = 0.4` reflects a plausible upper bound on porosity for
              most sands/silts; tighten further for site-specific priors.
- `regtype` : `:R0` (smallest), `:R1` (first-difference), or `:R2`
              (curvature) — see `makereg(::Symbol, ::SMRDetOperator)`.
              Default `:R1`. `:R2` leaves linear trends in the null space
              and is a poor choice for SMR (it produces large deep ramps).
- `anchor_top` : anchor the top cell towards zero in `:R1` (a weak
              minimum-norm pull, see `makereg`). Default `true`; `false`
              is the pure CPC87 roughness. The two differ little in
              practice.
- `λ²min`, `λ²max` : log10 bounds on the smoothness-weight line search.
                      Default `[-1, 6]`.
- `β²`     : reference-model (Marquardt) penalty weight on `‖w − wref‖²`,
             scaled by `λ²`. Default `0` (pure Occam). Use `≈ 0.01` to
             suppress spurious smooth water below the depth of resolution
             — see `makereg`.
- `knownvalue` : Gauss-Newton acceptance threshold — a step is accepted
                 if it drops χ² to at most `knownvalue · χ²₀`. Inherited
                 from HiQGA's `gradientinv`. Default `0.7`.
- `nstepsmax` : maximum outer iterations. Default `15`.
- `ntries` : line-search resolution on `λ²`. Default `6`.
- `target` : χ² target. Default `length(S.V0)` (χ²/N = 1).
- `verbose` : print `gradientinv`'s per-iteration log. Default `false`.
- `uncertainty` : one of
    - `:none` (default) — skip uncertainty estimation, return `nothing`
      in the `uncertainty` field.
    - `:laplace` or `:linearised` — compute the Hessian-inverse Gaussian
      approximation via `linearised_posterior_covariance` at the final
      `λ²`. Cheap (one Cholesky). Assumes Gaussian noise, near-linear
      forward around the minimum, and single-mode posterior. Also returns
      the model resolution matrix.
    - `:bootstrap` — parametric bootstrap via `bootstrap_invert`. Resamples
      noise and re-runs Occam per replicate; returns per-cell quantiles.
      Slower (N re-inversions) but captures non-linearity and
      per-replicate `λ²` sensitivity. Does NOT capture smoothing bias.
- `N_bootstrap` : number of bootstrap replicates when `uncertainty = :bootstrap`.
                  Default 200.
- `bootstrap_method` : `:parametric` (default, Gaussian noise from σ_V0)
                       or `:residual` (non-parametric residual resampling).
- `bootstrap_rseed`  : base RNG seed for reproducibility. Default `nothing`
                       (uses global RNG).
- Any remaining keyword arguments are forwarded to `gradientinv`.

Returns a NamedTuple with fields:
- `op`  : the `SMRDetOperator` — holds the final Jacobian, residual and weights.
- `mnew`, `χ²`, `λ²`, `oidx` : `gradientinv`'s per-iteration histories.
- `uncertainty` : `nothing`, or the result NamedTuple from
  `linearised_posterior_covariance` or `bootstrap_invert`.
- `w`, `χ²_final`, `λ²_final` : the final model, its χ² and its λ².
- `converged` : `true` if `χ²_final ≤ target · (1 + 1e-2)`. HiQGA's
  `gradientinv` silently relaxes the target when it cannot be reached within
  `nstepsmax` iterations and returns the smoothest model at the χ² it did
  reach, so always check this flag.

Positional unpacking of the first five fields still works:

    op, mnew, χ², λ², oidx = deterministic_invert(S; ...)

See also: `linearised_posterior_covariance`, `bootstrap_invert`.
"""
function deterministic_invert(S::SMRSoundingKnown;
        wstart = fill(0.05, length(S.Fm.zgrid)),
        wref   = fill(0.05, length(S.Fm.zgrid)),
        lo = 1e-4, hi = 0.4,
        regtype = :R1,
        anchor_top::Bool = true,
        λ²min = -1.0,
        λ²max = 6.0,
        β²    = 0.0,
        knownvalue = 0.7,
        nstepsmax = 15,
        ntries    = 6,
        target    = nothing,
        verbose   = false,
        uncertainty::Symbol = :none,
        N_bootstrap::Int = 200,
        bootstrap_method::Symbol = :parametric,
        bootstrap_rseed::Union{Nothing, Int} = nothing,
        kwargs...)
    op = SMRDetOperator(S; anchor_top = anchor_top)
    length(wstart) == size(op.K̃, 2) || throw(ArgumentError("wstart length must equal length(S.Fm.zgrid)"))
    length(wref)   == length(wstart) || throw(ArgumentError("wref length must equal wstart length"))
    mnew, χ², λ², oidx = transD_GP.gradientinv(
        collect(Float64, wstart), collect(Float64, wref), op;
        regtype, λ²min, λ²max, β², knownvalue, nstepsmax, ntries, target, lo, hi, verbose, kwargs...)

    w_final  = mnew[end][oidx[end]]
    χ²_final = χ²[end][oidx[end]]
    λ²_final = λ²[end][oidx[end]][1]
    target₀  = isnothing(target) ? length(S.V0) : target
    converged = χ²_final <= target₀ * (1 + 1e-2)
    # leave op.J / op.res at the final model (gradientinv's last evaluation may not be it)
    transD_GP.AbstractOperator.getresidual(op, w_final; computeJ = true)

    unc = _compute_uncertainty(uncertainty, S, op, mnew, λ², oidx;
        wstart = wstart, wref = wref, regtype = regtype, anchor_top = anchor_top,
        lo = lo, hi = hi, β² = β², λ²min = λ²min, λ²max = λ²max,
        knownvalue = knownvalue, nstepsmax = nstepsmax, ntries = ntries,
        target = target, verbose = verbose,
        N_bootstrap = N_bootstrap,
        bootstrap_method = bootstrap_method,
        bootstrap_rseed = bootstrap_rseed,
        kwargs...)

    return (op = op, mnew = mnew, χ² = χ², λ² = λ², oidx = oidx, uncertainty = unc,
            w = w_final, χ²_final = χ²_final, λ²_final = λ²_final, converged = converged)
end

# Dispatch helper — kept out of `deterministic_invert`'s body so the main
# function reads clearly and so future methods (e.g. jackknife, MCMC) can be
# slotted in by adding a branch here without disturbing the caller signature.
function _compute_uncertainty(kind::Symbol, S, op, mnew, λ², oidx;
        wstart, wref, regtype, anchor_top, lo, hi, β², λ²min, λ²max,
        knownvalue, nstepsmax, ntries, target, verbose,
        N_bootstrap, bootstrap_method, bootstrap_rseed, kwargs...)
    kind === :none && return nothing
    if kind === :laplace || kind === :linearised
        λ²_final = λ²[end][oidx[end]][1]
        return linearised_posterior_covariance(op, λ²_final; regtype = regtype, β² = β²)
    elseif kind === :bootstrap
        w0 = mnew[end][oidx[end]]
        # Forward the SAME regularisation setup to every bootstrap replicate.
        # Bootstrap characterises noise sensitivity around a fixed prior, not
        # sensitivity to the prior itself.
        return bootstrap_invert(S;
            N = N_bootstrap, w0 = w0,
            method = bootstrap_method,
            rseed = bootstrap_rseed,
            wstart = wstart, wref = wref, regtype = regtype, anchor_top = anchor_top,
            lo = lo, hi = hi, β² = β²,
            λ²min = λ²min, λ²max = λ²max,
            knownvalue = knownvalue, nstepsmax = nstepsmax, ntries = ntries,
            target = target, verbose = verbose,
            kwargs...)
    else
        throw(ArgumentError("uncertainty must be :none, :laplace (aka :linearised), or :bootstrap — got $kind"))
    end
end

"""
    linearised_posterior_covariance(op::SMRDetOperator, λ²::Real;
                                     regtype = :R1, β² = 0.0)
        -> NamedTuple with fields (Cm, σ, λ², R, diagR)

Compute the linearised Gaussian posterior covariance of the water-content
model around the point at which `op.J` was last populated.

The formula is the standard linear-inverse result (Tarantola 2005 eqs
3.37–3.40; Menke 2018 Ch. 5)

    C_m = (Jᵀ W² J + λ² · Rᵀ R + λ² β² I)⁻¹

where `J` is `op.J`, `W = op.W = diag(1 ./ σ_V₀)` (so `W² = op.W' * op.W` is
the inverse noise covariance), and `R` is the roughness matrix produced
by `HiQGA.transD_GP.makereg(regtype, op)`. This is the Gauss-Newton Hessian
used by `gradientinv`, so pass the same `regtype` and `β²`.

**Interpretation.** `C_m` is a posterior covariance only if `λ²RᵀR` is read
as a prior precision. For `:R1` that prior is a random walk in depth whose
variance grows linearly with depth, so where the data have no sensitivity
`σ` is the prior, not the data. The returned model resolution matrix

    R = C_m Jᵀ W² J          (Alumbaugh & Newman 2000; Menke 2018 §4.3)

tells you where that is: `σ` is only meaningful where `diag(R)` is close
to 1. Report the two together. `λ²` is the Occam line-search value, which
is a data-fit tuning parameter rather than a calibrated prior, so treat
the band as indicative. The covariance also ignores the `[lo, hi]` box
constraint used during inversion.

**`λ²` is deliberately mandatory.** Omitting the smoothness term
(`λ² = 0`) leaves `Jᵀ W² J`, which is ill-conditioned or singular for the
underdetermined smooth-model inversions this operator is designed for.

Returns a NamedTuple with:
- `Cm`    — full covariance matrix (dense, `nz × nz`).
- `σ`     — per-cell 1-σ uncertainty, `sqrt.(diag(Cm))`.
- `λ²`    — the smoothness weight used (echoed for downstream bookkeeping).
- `R`     — model resolution matrix (dense, `nz × nz`).
- `diagR` — its diagonal (1 = perfectly resolved by the data, 0 = set by the
  regulariser). `R` is not symmetric, so entries can dip slightly below zero
  where the regulariser dominates.
"""
function linearised_posterior_covariance(op::SMRDetOperator, λ²::Real;
                                          regtype::Symbol = :R1, β²::Real = 0.0)
    Rreg = transD_GP.makereg(regtype, op)
    JtW  = op.J' * op.W
    JtWWJ = Matrix(JtW * JtW')
    H    = JtWWJ .+ float(λ²) .* Matrix(Rreg' * Rreg) .+ float(λ² * β²) .* Matrix(1.0I, size(JtWWJ)...)
    Hs   = Symmetric(H)
    Cm   = cholesky(Hs; check = false)
    if issuccess(Cm)
        Cm = inv(Cm)
    else
        @warn "linearised_posterior_covariance: Hessian not positive definite; using symmetric pseudo-inverse"
        Cm = pinv(Hs)
    end
    σ    = sqrt.(max.(diag(Cm), 0.0))
    Rres = Cm * JtWWJ
    return (Cm = Cm, σ = σ, λ² = float(λ²), R = Rres, diagR = diag(Rres))
end
