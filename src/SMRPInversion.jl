# wrapper to integrate SMRPI 1D forward model with
# HiQGA/transD_GP
#
# References
# ----------
# * Surface NMR forward kernel, co-/counter-rotating field decomposition and
#   the complex sounding curve: Weichman, Lavely & Ritzwoller (2000), Phys.
#   Rev. E 62, 1290–1312; Hertrich (2008), Prog. NMR Spectrosc. 53, 227–248;
#   Legchenko & Valla (2002), J. Appl. Geophys. 50, 3–19. Kernel implemented
#   in SNMRForward.jl.
# * Trans-dimensional Gaussian-process prior and parallel-tempered MCMC used
#   through HiQGA: Ray & Myer (2019), GJI 217, 1706–1726; Ray, Alumbaugh,
#   Hoversten & Key (2013), Geophysics 78, E271–E280.
# * Maximum-likelihood noise scaling (`SMRSoundingUnknown`): the profiled
#   likelihood ½N·log(rᵀr) follows from integrating a scalar variance out
#   under a Jeffreys prior — see Malinverno & Briggs (2004), Geophysics 69,
#   1005–1016. The hierarchical alternative that samples the scale is in
#   HierarchicalNoise.jl.
# * Loop convention: `MRSForward_square(L, ...)` takes the HALF-side `L`
#   (a 100 m square loop is `L = 50`). Verified numerically against a
#   circular loop of equal area in test/test_deterministic.jl.
module SMRPInversion
using HiQGA.transD_GP.AbstractOperator, HiQGA.transD_GP.CommonToAll
import HiQGA.transD_GP.AbstractOperator.get_misfit
import HiQGA.transD_GP.Model, HiQGA.transD_GP.Options
import HiQGA.transD_GP.ModelNuisance, HiQGA.transD_GP.OptionsNuisance
import HiQGA.transD_GP.AbstractOperator.Sounding
using HiQGA.transD_GP
using SNMRForward, Random, PyPlot

using Distributed, Dates

export γh, ConductivityModel, MRSForward, MRSForward_square
export newSMRSounding, create_synthetic

include("ProcessingTools.jl")
using .ProcessingTools

abstract type SMRSounding <: Operator1D end

mutable struct SMRSoundingKnown <: SMRSounding
    V0 :: Vector{<:Real} #sounding curve
    ϕ :: Vector{<:Real} #phases
    σ_V0 :: Vector{<:Real}
    σ_ϕ :: Vector{<:Real}
    Fm :: SNMRForward.MRSForward
    linearsat :: Bool
    amponly :: Bool
    offset_ϕ :: Real
    stretch_ϕ ::Real 
end

mutable struct SMRSoundingUnknown <: SMRSounding
    # this struct doesn't contain noise variances
    # - a maximum likelihood estimate is computed by the sampler instead
    V0 :: Vector{<:Real}
    ϕ :: Vector{<:Real}
    Fm :: SNMRForward.MRSForward
    σd_diag :: Vector{<:Real}
    linearsat :: Bool
    amponly :: Bool
    offset_ϕ :: Real
    stretch_ϕ :: Real
end

function newSMRSounding(V0, ϕ, Fm; σ_V0=nothing, σ_ϕ=nothing, mult=false, linearsat=true, amponly=true, showplot=false,
                        offset_ϕ = 0., stretch_ϕ=1.)
    (length(V0) != length(ϕ) && 
        throw(ArgumentError("V0 and ϕ must have same length")))
    (amponly && (!isapprox(offset_ϕ, 0.) | !isapprox(stretch_ϕ, 1.))) &&
        throw(ArgumentError("Amplitude only requires no phase modification"))
    if !isnothing(σ_V0) || !isnothing(σ_ϕ)
        if isnothing(σ_V0) || isnothing(σ_ϕ)
            throw(ArgumentError("σ_V0 and σ_ϕ must both be provided, or neither."))
        end
        if length(V0) != length(σ_V0) || length(ϕ) != length(σ_ϕ)
            throw(ArgumentError("σ_V0 and σ_ϕ must have the same length as the associated data"))
        end
        S = SMRSoundingKnown(V0, ϕ, σ_V0, σ_ϕ, Fm, linearsat, amponly, offset_ϕ, stretch_ϕ)
        return S
    end
    if mult
        σd_diag = [V0; ones(length(ϕ))]
    else
        σd_diag = [ones(length(V0)); 1 ./ V0]
    end
    S = SMRSoundingUnknown(V0, ϕ, Fm, σd_diag, linearsat, amponly, offset_ϕ, stretch_ϕ)
    showplot && plotdata(S)
    S
end

function extractnu!(S::SMRSounding, nuvec::AbstractVector) # useful for plotting reuse
    S.offset_ϕ = nuvec[1]
    S.stretch_ϕ = 10. ^nuvec[2]
end    

phasemodif(ϕ; stretch=1, offset=0) = stretch*ϕ .+ offset 

function applynu(S::SMRSounding, w::Vector{<:Real}) # useful for plotting and misfit reapply
    applynu(S.Fm, w, offset_ϕ=S.offset_ϕ, stretch_ϕ=S.stretch_ϕ)
end

function applynu(Fm::SNMRForward.MRSForward, w::AbstractVector; offset_ϕ=0., stretch_ϕ=1.)
    response = SNMRForward.forward(Fm, w)
    Vres = abs.(response)
    ϕres = angle.(response)
    ϕres = phasemodif(ϕres, offset=offset_ϕ, stretch=stretch_ϕ)    
    Vres, ϕres
end

function get_misfit(m::Model, opt::Options, S::SMRSounding)
    opt.debug && return 0.0
    get_misfit(S.linearsat ? m.fstar[:] : 10 .^ m.fstar[:], S)
end
# above defined function and type signature MUST be defined

function get_misfit(m::Model, mn::ModelNuisance, opt::Union{Options,OptionsNuisance}, S::SMRSounding)
    # the "nuisance" in this case is a constant offset phase and log10 of a stretch factor
    opt.debug && return 0.0
    extractnu!(S, mn.nuisance)
    get_misfit(S.linearsat ? m.fstar[:] : 10 .^ m.fstar[:], S, )
end

function get_misfit(w::Vector{<:Real}, S::SMRSounding)
    Vres, ϕres = applynu(S, w)
    if isa(S, SMRSoundingKnown)
        #use provided noise
        residual = (S.V0 .- Vres)./S.σ_V0
        S.amponly || (residual = [residual; rem2pi.(S.ϕ - ϕres, RoundNearest)./S.σ_ϕ])
        return residual' * residual / 2
    else
        # maximum-likelihood noise models
        residual = (S.V0 - Vres)./S.σd_diag[1:length(Vres)]
        S.amponly || (residual = [residual; rem2pi.(S.ϕ - ϕres, RoundNearest)./S.σd_diag[length(Vres)+1:end]])
        return length(residual)/2 * log(residual' * residual)
    end
end

"""
    create_synthetic(w, σ, t, Be, ϕ, R, zgrid, qgrid; kwargs...) -> SMRSounding

Forward-model a water-content profile `w` (nodal values on `zgrid`) over the
layered conductivity model `(σ, t)` and add noise.

- `R`      : loop size. For `square = true` (the default) this is the
             **half-side** of a square loop (100 m loop → `R = 50`); for
             `square = false` it is the radius of a circular loop.
- `ϕ`      : Earth-field inclination (rad); `θ`: loop azimuth from magnetic
             north (rad, square loop only); `Be`: Earth field (T).
- `noise_domain` : `:amplitude` (default) adds independent Gaussian noise to
             |V| and to the phase (σ_ϕ = σ_V/|V|), the high-SNR limit.
             `:complex` adds Gaussian noise of standard deviation `σ_V` to
             each quadrature of the complex response and takes modulus and
             angle, which gives the correct Rician amplitude statistics at
             low SNR (Rice 1944; bias ≈ +14 % at SNR 2).
- `mult`   : multiplicative (`noise_frac`) rather than additive
             (`noise_additive`, volts) noise.
- `noise_mle` : return an `SMRSoundingUnknown` (noise level profiled by ML)
             instead of an `SMRSoundingKnown`.
- other keywords as `newSMRSounding`.
"""
function create_synthetic(w::Vector{<:Real}, σ::Vector{<:Real}, t::Vector{<:Real},
            Be::Real, ϕ::Real, R::Real, zgrid::Vector{<:Real}, qgrid::Vector{<:Real}
    ; noise_frac = 0.05, noise_additive = 100e-9, θ = 0., square=true, noise_mle = false, mult = false, linearsat=false,
    amponly=false, offset_ϕ = 0., stretch_ϕ = 1., showplot=true, rseed=131, noise_domain::Symbol = :amplitude)
    Random.seed!(rseed)
    ct = SNMRForward.ConductivityModel(σ, t)
    noise_domain in (:amplitude, :complex) ||
        throw(ArgumentError("noise_domain must be :amplitude or :complex"))

    # NOTE: before Oct 2026 this ternary was inverted (square=false gave the
    # square loop). The default is now square=true so existing call sites that
    # relied on the default keep getting a square loop.
    F = (square ?
        SNMRForward.MRSForward_square(R, zgrid, qgrid, ϕ, θ, Be, ct) :
        SNMRForward.MRSForward(R, zgrid, qgrid, ϕ, Be, ct))

    Vres, ϕres = applynu(F,w, offset_ϕ=offset_ϕ, stretch_ϕ=stretch_ϕ)
    if mult
        σ_V0 = noise_frac * Vres
        σ_ϕ = noise_frac * ones(size(Vres))
    else
        σ_V0 = noise_additive * ones(size(Vres))
        σ_ϕ = noise_additive ./ Vres
    end
    if noise_domain === :complex
        resp = Vres .* exp.(im .* ϕres) .+ σ_V0 .* (randn(size(Vres)) .+ im .* randn(size(Vres)))
        noisy_V0 = abs.(resp)
        noisy_ϕ = angle.(resp)
    else
        noisy_V0 = Vres .+ σ_V0 .* randn(size(Vres))
        noisy_ϕ = ϕres .+ σ_ϕ .* randn(size(Vres))
    end

    if noise_mle
        σ_V0 = nothing
        σ_ϕ = nothing
    end
    S = newSMRSounding(noisy_V0, noisy_ϕ, F, σ_V0=σ_V0, σ_ϕ=σ_ϕ, mult=mult, linearsat=linearsat, 
                    amponly=amponly, offset_ϕ=offset_ϕ, stretch_ϕ=stretch_ϕ)
    if showplot 
        fig = plotmodelcurve(ct.σ, ct.d, w, zgrid, Vres, ϕres, qgrid, modelalpha=1)
        plotdata(S, fig, iaxis=3, writelabel=false, msize=8)
    end
    S
end

## plotting stuff

function plotdata(S::SMRSounding; gridalpha=0.5, figsize=(6,3), msize=8)
    fig, ax = plt.subplots(1, 2, sharex=true, figsize=figsize)
    plotdata(S, fig, gridalpha=gridalpha, msize=msize)
    fig 
end

function plotdata(S::SMRSounding, fig; iaxis=1, gridalpha=0.5, writelabel=true, msize=8)
    ax = fig.axes
    ϕ = rem2pi.(S.ϕ, RoundNearest)
    if isa(S, SMRSoundingKnown)
        σ_ϕ = rem2pi.(S.σ_ϕ, RoundNearest)
        ax[iaxis].errorbar(S.Fm.qgrid, S.V0, S.σ_V0, linestyle="none", marker=".", elinewidth=1, capsize=3)
        ax[iaxis+1].errorbar(S.Fm.qgrid, ϕ, σ_ϕ, linestyle="none", marker=".", elinewidth=1, capsize=3)
    else
        ax[iaxis].plot(S.Fm.qgrid, S.V0, linestyle="none", marker="o", markersize=msize)
        ax[iaxis].plot(S.Fm.qgrid, S.V0, linestyle="none", marker=".", markersize=msize/2)
        ax[iaxis+1].plot(S.Fm.qgrid, ϕ, linestyle="none", marker="o", markersize=msize)
        ax[iaxis+1].plot(S.Fm.qgrid, ϕ, linestyle="none", marker=".", markersize=msize/2)
    end
    if writelabel # useful on the last iteration of model plotting
        writelabels(ax, iaxis, gridalpha)
        ax[iaxis+1].set_ylim(-pi, pi)
    end 
    fig.tight_layout()
end    

function plotmodelcurve(c, t, w, z, V0, ϕ, q; gridalpha=0.5, modelalpha=0.5,figsize=(10,3),
    lcolor="nocolor", writelabel=true)
    # conductivity and saturation with depth initialize
    fig = figure(figsize=(figsize))
    s1 = subplot(141)
    s2 = subplot(142, sharey=s1)
    s3 = subplot(143)
    s4 = subplot(144, sharex=s3)
    plotmodelcurve(c, t, w, z, V0, ϕ, q, fig, gridalpha=gridalpha, modelalpha=modelalpha, writelabel=writelabel, lcolor=lcolor)
    fig
end

function plotmodelcurve(c, t, w, z, V0, ϕ, q, fig; gridalpha=0.5, modelalpha=0.5, writelabel=true, lcolor="nocolor")
    # conductivity and saturation into axis
    ax = fig.axes
    zfromt = [0.; cumsum(t)]
    if isempty(t) 
        push!(zfromt, maximum(z))
        cuse = [c;c[end]]
    else
        cuse = c
    end
    ax[1].step(cuse, zfromt)
    ax[1].set_xscale("log")
    if writelabel
        ax[1].grid(true, which="both", alpha=gridalpha)
        ax[1].set_xlabel("conductivity S/m")
        ax[1].set_ylim(reverse(extrema(z)))
    end
    plotmodelcurve(w, z, V0, ϕ, q, fig; iaxis=2, gridalpha=gridalpha, modelalpha=modelalpha, writelabel=writelabel, lcolor=lcolor,)
end    

function plotmodelcurve(w, z, V0, ϕ, q; gridalpha=0.5, modelalpha=0.5,figsize=(10,3),
    lcolor="nocolor", writelabel=true)
    # saturation with depth initialize
    fig = figure(figsize=(figsize))
    s1 = subplot(131)
    s2 = subplot(132)
    s3 = subplot(133, sharex=s2)
    plotmodelcurve(w, z, V0, ϕ, q, fig, gridalpha=gridalpha, modelalpha=modelalpha, writelabel=writelabel, lcolor=lcolor)
    fig
end

function plotmodelcurve(w, z, V0, ϕ, q, fig; iaxis=1, gridalpha=0.5, modelalpha=0.5, writelabel=true, lcolor="nocolor")
    # saturation with depth into axis
    ax = fig.axes
    if lcolor == "nocolor"
        ax[iaxis].step(w, z, alpha=modelalpha)
    else    
        ax[iaxis].step(w, z, color=lcolor, alpha=modelalpha)
    end    
    if writelabel
        ax[iaxis].grid(true, which="both", alpha=gridalpha)
        ax[iaxis].set_xlabel("saturation")
        ax[iaxis].set_ylabel("depth m")
        ax[iaxis].set_ylim(reverse(extrema(z)))
    end    
    plotcurve(V0, ϕ, q, fig, iaxis=iaxis+1, gridalpha=gridalpha, modelalpha=modelalpha,
                    lcolor=lcolor, writelabel=writelabel)
end

function plotcurve(V0, ϕ, q; gridalpha=0.5)
    # sometimes you only want to plot the responses, as in for a paper
    fig, ax = plt.subplots(1, 2, sharex=true)
    plotcurve(V0, ϕ, q, fig, gridalpha=gridalpha)
    fig
end   

function plotcurve(V0, ϕ, q, fig; iaxis=1, gridalpha=0.5, modelalpha=0.5, 
    lcolor="nocolor", writelabel=true)
    # plotting responses into an existing figure, useful for plotting multiple responses
    ax = fig.axes
    if lcolor == "nocolor" # as in use default PyPlot colors
        ax[iaxis].plot(q, V0)
        ax[iaxis+1].plot(q, ϕ)
    else # plot with specified color
        ax[iaxis].plot(q, V0, color=lcolor, alpha=modelalpha)
        ax[iaxis+1].plot(q, rem2pi.(ϕ, RoundNearest), color=lcolor, alpha=modelalpha)
    end
    if writelabel # useful on the last iteration of model plotting
        writelabels(ax, iaxis, gridalpha)
        ax[iaxis+1].set_ylim(-pi, pi)
    end    
    fig.tight_layout()
    nothing
end  

function writelabels(ax, iaxis, gridalpha)
    ax[iaxis].set_xlabel("Pulse moment A-s")
    ax[iaxis].set_ylabel("Amplitude")
    ax[iaxis].set_xscale("log")
    ax[iaxis].grid(true, which="both", alpha=gridalpha)
    ax[iaxis+1].set_xlabel("Pulse moment A-s")
    ax[iaxis+1].set_ylabel("phase")
    ax[iaxis+1].grid(true, which="both", alpha=gridalpha)
end  

function plot_model_field(S::SMRSounding, opt::Options, optn::OptionsNuisance; 
        gridalpha=0.5, figsize=(10,4), lcolor="nocolor", modelalpha=0.5, burninfrac=0.5, 
        decfactor=10)
    fig = initfig(figsize)
    M = assembleTat1(opt, :fstar, temperaturenum=1, burninfrac=burninfrac)[1:decfactor:end]
    Mnu = assemblenuisancesatT(optn, burninfrac = burninfrac, temperaturenum = 1)[1:decfactor:end,:]
    @assert length(M) == size(Mnu, 1)
    T = S.linearsat ? x->x : x->10^x
    for (imodel, m) in enumerate(M)
        w = T.(m)[:]
        extractnu!(S, vec(Mnu[imodel,:]))
        Vresp, ϕresp = applynu(S, w)
        writelabel = imodel == length(M) ? true : false
        plotmodelcurve(w, S.Fm.zgrid, Vresp, ϕresp, S.Fm.qgrid, fig, writelabel=writelabel,
            gridalpha=gridalpha, lcolor=lcolor, modelalpha=modelalpha)
    end
    plotdata(S, fig, iaxis=2, gridalpha=gridalpha)    
end  

function plot_model_field(S::SMRSounding, opt::Options;
        gridalpha=0.5, figsize=(10,4), lcolor="nocolor", modelalpha=0.5, burninfrac=0.5, 
        decfactor=10)
    fig = initfig(figsize)
    M = assembleTat1(opt, :fstar, temperaturenum=1, burninfrac=burninfrac)[1:decfactor:end]
    T = S.linearsat ? x->x : x->10^x
    for (imodel, m) in enumerate(M)
        w = T.(m)[:]
        @assert (isapprox(S.stretch_ϕ, 1.) && isapprox(S.offset_ϕ, 0.))
        Vresp, ϕresp = applynu(S, w)  
        writelabel = imodel == length(M) ? true : false
        plotmodelcurve(w, S.Fm.zgrid, Vresp, ϕresp, S.Fm.qgrid, fig, writelabel=writelabel,
            gridalpha=gridalpha, lcolor=lcolor, modelalpha=modelalpha)
    end
    plotdata(S, fig, iaxis=2, gridalpha=gridalpha)    
end  

function initfig(figsize)
    fig = figure(figsize=(figsize))
    s1 = subplot(131)
    s2 = subplot(132)
    s3 = subplot(133, sharex=s2)
    fig
end

mutable struct SMRSoundingWrapper <: Sounding
    sounding_name :: String
    sounding :: SMRSounding
end

function loopacrosssoundings(
    soundings::Array{S, 1};
    nsequentialiters   = -1,
    nparallelsoundings = -1,
    Tmax               = -1,
    nsamples           = -1,
    nchainsatone       = -1,
    nchainspersounding = -1,
    nmin               = 2,
    nmax               = 40,
    K                  = GP.OrstUhn(),
    demean             = false,
    sampledc           = true,
    sddc               = 0.01,
    sdpos              = 0.05,
    sdprop             = 0.05,
    fbounds            = [-0.5 2.5],
    xbounds            = [],
    xall               = [],
    λ                  = [2],
    δ                  = 0.01,
    save_freq          = 50,
    nuisance_sdev      = [0., 0.],
    nuisance_bounds    = [0. 0.;
                            0. 0.],
    updatenuisances    = true,
) where S<:Sounding

    @assert nsequentialiters  != -1
    @assert nparallelsoundings != -1
    @assert nchainspersounding != -1
    @assert nsamples != - 1
    @assert nchainsatone != -1
    @assert Tmax != -1
    @assert length(xbounds) > 0
    @assert length(xall) > 0

    nsoundings = length(soundings)

    for iter = 1:nsequentialiters
        if iter<nsequentialiters
            ss = (iter-1)*nparallelsoundings+1:iter*nparallelsoundings
        else
            ss = (iter-1)*nparallelsoundings+1:nsoundings
        end
        @info "soundings in loop $iter of $nsequentialiters", ss
        r_nothing = Array{Nothing, 1}(undef, length(ss))
        @sync for (i, s) in Iterators.reverse(enumerate(ss))
            pids = (i-1)*nchainspersounding+i:i*(nchainspersounding+1)
            @info "pids in sounding $s:", pids

            opt = transD_GP.OptionsStat(
                nmin = nmin,
                nmax = nmax,
                xbounds = xbounds,
                fbounds = fbounds,
                fdataname = soundings[s].sounding_name*"_",
                xall = xall,
                λ = λ,
                δ = δ,
                sdev_prop = sdprop,
                sdev_pos = sdpos,
                save_freq = save_freq,
                demean = demean,
                sampledc = sampledc,
                sdev_dc = sddc,
                quasimultid = false,
                K = K,
            )
            
            optn = transD_GP.OptionsNuisance(
                opt;
                sdev = nuisance_sdev,
                bounds = nuisance_bounds,
                updatenuisances = updatenuisances
            )

            @async r_nothing[i] = remotecall_fetch(transD_GP.main, pids[1], opt, optn, soundings[s].sounding, collect(pids[2:end]),
                                    Tmax         = Tmax,
                                    nsamples     = nsamples,
                                    nchainsatone = nchainsatone)

        end # @sync
        @info "done $iter out of $nsequentialiters at $(Dates.now())"
    end
end

# ── Deterministic (Occam-style Gauss-Newton) inversion ──────────────────────
# Amplitude-only Occam wrapper around HiQGA.transD_GP.gradientinv.
# Defined here so SMRSoundingKnown (from earlier in this file) is in scope.
include("DeterministicInversion.jl")

# ── Hierarchical heteroscedastic noise for trans-D MCMC (Paper 1 centrepiece)
include("HierarchicalNoise.jl")

# ── Bootstrap uncertainty for the Occam inversion above ─────────────────────
# Parametric or non-parametric bootstrap; produces an ensemble of Occam
# solutions comparable to linearised_posterior_covariance and MCMC posteriors.
include("BootstrapInversion.jl")

end
