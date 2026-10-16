# Bootstrap uncertainty for Occam-style deterministic SMR inversion.
#
# Two flavours:
#   :parametric  — resample fresh Gaussian noise from the asserted σ_V0 and
#                  add to the noise-free forward at the best-fit w0. Standard
#                  parametric bootstrap; matches the noise model deterministic
#                  Occam and MCMC already assume.
#   :residual    — resample the observed standardised residuals
#                  (V0 - |K̃·w0|)/σ_V0 with replacement and rescale by σ_V0.
#                  Non-parametric; useful when the noise distribution is
#                  uncertain and you have field data.
#
# References
# ----------
# * Efron (1979), Ann. Statist. 7, 1–26; Efron & Tibshirani (1993), An
#   Introduction to the Bootstrap, Chapman & Hall, Ch. 6 (parametric) and
#   Ch. 9 (residual resampling in regression).
# * Geophysical usage: Tichelaar & Ruff (1989), Eos 70, 593–606;
#   Aster, Borchers & Thurber (2018) §11.5.
#
# Both produce an ensemble of Occam solutions. Their per-depth spread is a
# non-linear, non-Gaussian uncertainty band directly comparable to
# `linearised_posterior_covariance` (the Gaussian ±2σ approximation) and to
# an MCMC p5-p95 credible interval. Where the three agree, the linearised
# assumption is safe; where bootstrap is wider, non-linearity or multi-modality
# matters and the linearised band is over-confident.
#
# What bootstrap does NOT capture: regularisation (smoothing) bias. Every
# replicate is a smoothed Occam solution centred on the smoothed `w0`, so
# where the data have little resolution the band will be narrow and will
# exclude the truth. Check `diagR` from `linearised_posterior_covariance`
# or compare against a known synthetic before reading the band as a
# credible interval.
#
# Cost: one Occam per replicate. Occam converges in a few seconds per SMR
# sounding, so N = 100-200 replicates typically fits in a few minutes.

using LinearAlgebra, SparseArrays, Random, Statistics

export bootstrap_invert

"""
    bootstrap_invert(S::SMRSoundingKnown;
                     N = 200,
                     w0 = nothing,
                     method = :parametric,
                     noise_domain = :amplitude,
                     rseed = nothing,
                     verbose = false,
                     kwargs...) -> NamedTuple

Bootstrap uncertainty for `deterministic_invert`.

Arguments:
- `S`      : sounding with known noise (`SMRSoundingKnown`).
- `N`      : number of bootstrap replicates. Default 200.
- `w0`     : centre model. Required for `:parametric` (noise is added to the
             forward at `w0`) and for `:residual` (residuals are computed
             against the forward at `w0`). If `nothing` (default), first
             solves the unperturbed problem with `deterministic_invert` and
             uses that solution. Pass a pre-computed best fit to save one
             inversion.
- `method` : `:parametric` — add fresh Gaussian noise ∼ N(0, σ_V0²) per
             replicate. Matches the noise model MCMC / linearised assume.
             `:residual` — draw the observed standardised residuals
             `(V0 - |K̃·w0|)/σ_V0` with replacement, rescale by `σ_V0`
             (non-parametric, handles the heteroscedastic gates correctly).
             Default `:parametric`.
- `noise_domain` : `:amplitude` (default) adds Gaussian noise to `|K̃·w0|`;
             `:complex` adds Gaussian noise of standard deviation `σ_V0` to
             each quadrature of the complex forward `K̃·w0` and takes the
             modulus, which gives the physically correct Rician amplitude
             statistics at low SNR (Rice 1944). Only for `:parametric`.
- `rseed`  : base seed for reproducibility. If given, replicate `i` uses
             `MersenneTwister(rseed + i)`. If `nothing`, uses the global RNG.
- `verbose`: log per-replicate progress. Default `false`. HiQGA's own
             per-iteration log is always switched off inside the replicates.
- `kwargs...` : forwarded to `deterministic_invert` (`wstart`, `wref`,
                `regtype`, `anchor_top`, `β²`, `lo`, `hi`, `λ²min`, `λ²max`,
                `target`, etc.).
                **Critical**: pass the SAME `wstart` / `wref` / `regtype` /
                `β²` you used for the original inversion. Bootstrap
                characterises noise sensitivity around a fixed regularisation
                choice, not sensitivity to the prior itself.

Returns a NamedTuple with:
- `w0`      : centre model used (either supplied or from the initial fit).
- `w_reps`  : `(nz × N)` matrix of replicate Occam solutions. Columns
              corresponding to failed inversions are all `NaN`.
- `n_ok`    : number of replicates that ran to completion.
- `converged_reps` : `(N,)` `Bool`, whether each replicate reached its χ²
              target (see `deterministic_invert`'s `converged`).
- `n_converged` : number of replicates that both ran and reached the target.
- `q05, q50, q95` : per-cell quantiles across successful replicates.
- `mean_w, std_w` : per-cell mean and standard deviation across replicates.
- `λ²_reps` : final `λ²` chosen by each replicate's line search — spread
              here is the regularisation-weight sensitivity the linearised
              approach implicitly misses.
- `χ²_reps` : final `χ²` achieved by each replicate.
- `method`, `noise_domain` : echoed.

Notes:
- Bootstrap does not capture `wref` uncertainty (each replicate uses the
  same reference model) nor smoothing bias (see module header).
- For `:amplitude` noise, perturbations that push individual gates negative
  are clipped to a tiny positive floor before inversion, as negative
  amplitudes are unphysical; `:complex` noise never needs clipping.
"""
function bootstrap_invert(S::SMRSoundingKnown;
        N::Int = 200,
        w0::Union{Nothing, AbstractVector} = nothing,
        method::Symbol = :parametric,
        noise_domain::Symbol = :amplitude,
        rseed::Union{Nothing, Int} = nothing,
        verbose::Bool = false,
        kwargs...)

    method in (:parametric, :residual) ||
        throw(ArgumentError("method must be :parametric or :residual, got $method"))
    noise_domain in (:amplitude, :complex) ||
        throw(ArgumentError("noise_domain must be :amplitude or :complex, got $noise_domain"))

    # HiQGA's gradientinv log is off in the replicates unless the caller insists
    kw = merge((; kwargs...), (verbose = false,))

    # Solve the unperturbed problem if the caller didn't supply a centre model
    if w0 === nothing
        verbose && @info "bootstrap_invert: solving unperturbed Occam for w0"
        r0 = deterministic_invert(S; kw...)
        w0 = r0.w
    end
    w0 = collect(Float64, w0)

    # Noise-free forward at w0 via the amplitude operator.
    op_ref = SMRDetOperator(S)
    length(w0) == size(op_ref.K̃, 2) ||
        throw(ArgumentError("w0 length ($(length(w0))) must equal kernel column count ($(size(op_ref.K̃, 2)))"))
    V_cplx = op_ref.K̃ * w0
    V_hat  = abs.(V_cplx)

    # Standardised residuals only needed for :residual method
    zres = method === :residual ? (S.V0 .- V_hat) ./ S.σ_V0 : Float64[]

    nz = length(w0)
    w_reps  = fill(NaN, nz, N)
    λ²_reps = fill(NaN, N)
    χ²_reps = fill(NaN, N)
    converged_reps = falses(N)
    n_ok = 0

    for i in 1:N
        rng = rseed === nothing ? Random.GLOBAL_RNG : MersenneTwister(rseed + i)

        # Generate a bootstrap V0
        if method === :parametric
            if noise_domain === :complex
                n_c = S.σ_V0 .* (randn(rng, length(V_hat)) .+ im .* randn(rng, length(V_hat)))
                V_boot = abs.(V_cplx .+ n_c)
            else
                V_boot = max.(V_hat .+ randn(rng, length(V_hat)) .* S.σ_V0, 1e-30)
            end
        else
            idx = rand(rng, 1:length(zres), length(zres))
            V_boot = max.(V_hat .+ zres[idx] .* S.σ_V0, 1e-30)
        end

        # Build the bootstrap sounding — swap only V0, keep everything else
        S_boot = SMRSoundingKnown(V_boot, S.ϕ, S.σ_V0, S.σ_ϕ, S.Fm,
                                   S.linearsat, S.amponly, S.offset_ϕ, S.stretch_ϕ)

        try
            r = deterministic_invert(S_boot; kw...)
            w_reps[:, i] = r.w
            λ²_reps[i]   = r.λ²_final
            χ²_reps[i]   = r.χ²_final
            converged_reps[i] = r.converged
            n_ok += 1
            verbose && (i % 25 == 0) && @info "bootstrap_invert: replicate $i / $N done ($n_ok OK)"
        catch e
            verbose && @warn "bootstrap replicate $i failed: $e"
        end
    end

    # Per-cell quantiles + mean/std, robust to NaN rows
    q05    = _rowwise(w_reps, x -> quantile(x, 0.05))
    q50    = _rowwise(w_reps, x -> quantile(x, 0.50))
    q95    = _rowwise(w_reps, x -> quantile(x, 0.95))
    mean_w = _rowwise(w_reps, mean)
    std_w  = _rowwise(w_reps, x -> length(x) > 1 ? std(x) : 0.0)

    return (w0 = w0,
            w_reps  = w_reps,
            n_ok    = n_ok,
            converged_reps = converged_reps,
            n_converged = count(converged_reps),
            q05     = q05, q50 = q50, q95 = q95,
            mean_w  = mean_w, std_w = std_w,
            λ²_reps = λ²_reps, χ²_reps = χ²_reps,
            method  = method, noise_domain = noise_domain)
end

# NaN-robust per-row reduction: apply `f` to the non-NaN entries of each row,
# returning NaN for all-NaN rows. Used for the summary stats above.
function _rowwise(mat::AbstractMatrix{Float64}, f)
    nz = size(mat, 1)
    out = Vector{Float64}(undef, nz)
    @inbounds for i in 1:nz
        row = filter(!isnan, view(mat, i, :))
        out[i] = isempty(row) ? NaN : f(row)
    end
    out
end
