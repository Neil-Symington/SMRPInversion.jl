# SMRPInversion.jl
SMRPI is a package that "glues" the surface NMR forward modelling code in [SNMRForward.jl](https://github.com/richardt94/SNMRForward.jl) to the general-purpose Bayesian inference platform in [HiQGA](https://github.com/GeoscienceAustralia/HiQGA.jl).

## What is here

| module | what it does |
|---|---|
| `src/SMRPInversion.jl` | `SMRSoundingKnown` / `SMRSoundingUnknown` operators for HiQGA's trans-dimensional MCMC, synthetic generation (`create_synthetic`), plotting |
| `src/DeterministicInversion.jl` | Occam-style Gauss-Newton inversion of amplitude-only data (`deterministic_invert`), with the linearised posterior covariance and model-resolution matrix (`linearised_posterior_covariance`) |
| `src/BootstrapInversion.jl` | parametric and residual bootstrap around the Occam solution (`bootstrap_invert`) |
| `src/HierarchicalNoise.jl` | `SMRSoundingHierarchical`: per-gate noise shape fixed, scalar noise scale `s` sampled jointly with the model as a HiQGA nuisance parameter |
| `src/ProcessingTools.jl` | sounding-curve extraction from FID records |

Conventions worth knowing:

- `MRSForward_square(L, ...)` and `create_synthetic(...; square = true)` take the **half-side** `L` of a square loop (a 100 m loop is `L = 50`). Verified in `test/test_deterministic.jl` against a circular loop of equal area.
- Water content is a nodal field on `zgrid` integrated with the trapezoidal rule (as in `SNMRForward.forward`). The first node is wherever you put it; `setupz(1.0, ...)` starts at 1 m, so the top metre carries no water.
- Amplitude-only likelihoods are Gaussian in |V|, the high-SNR limit of the Rician amplitude distribution. `create_synthetic(...; noise_domain = :complex)` and `bootstrap_invert(...; noise_domain = :complex)` generate noise in the complex domain if you need the low-SNR behaviour.
- `deterministic_invert` returns `converged`; HiQGA's `gradientinv` silently relaxes the χ² target when it cannot reach it, so check the flag.
- Occam can park a smooth ramp of water below the depth of resolution (χ² = N is still met; ∫w dz can be several times too large). Report `diagR` and ∫w dz, and use `β² ≈ 0.01` towards `wref` when it happens. `examples/occam_uncertainty.jl` shows the effect.

## Examples

```
julia --project=. examples/occam_uncertainty.jl         # Occam + Laplace band + resolution + bootstrap, ~1–2 min
julia --project=. examples/hierarchical_noise_mcmc.jl   # trans-D MCMC with sampled noise scale, ~5 min, 4 workers
julia --project=. examples/compare_occam_mcmc.jl        # Occam vs trans-D MCMC on one synthetic
julia --project=. examples/occam_on_udf_data.jl         # Occam on a UDF-format GMR sounding (needs data)
```

The `synthetic/`, `synthetic_nuisance/`, `example_sounding/`, `UDF_example/` and `invert_real_data/` directories are the original numbered MCMC workflows.

## Tests

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

## References

- Constable, S.C., Parker, R.L. & Constable, C.G. (1987). Occam's inversion: A practical algorithm for generating smooth models from electromagnetic sounding data. *Geophysics* 52, 289–300.
- Tarantola, A. (2005). *Inverse Problem Theory and Methods for Model Parameter Estimation*. SIAM.
- Alumbaugh, D.L. & Newman, G.A. (2000). Image appraisal for 2-D and 3-D electromagnetic inversion. *Geophysics* 65, 1455–1467.
- Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Malinverno, A. & Briggs, V.A. (2004). Expanded uncertainty quantification in inverse problems: Hierarchical Bayes and empirical Bayes. *Geophysics* 69, 1005–1016.
- Malinverno, A. & Parker, R.L. (2006). Two ways to quantify uncertainty in geophysical inverse problems. *Geophysics* 71, W15–W27.
- Bodin, T. et al. (2012). Transdimensional inversion of receiver functions and surface wave dispersion. *JGR* 117, B02301.
- Ray, A. & Myer, D. (2019). Bayesian geophysical inversion with trans-dimensional Gaussian process machine learning. *GJI* 217, 1706–1726.
- Weichman, P.B., Lavely, E.M. & Ritzwoller, M.H. (2000). Theory of surface nuclear magnetic resonance with applications to geophysical imaging problems. *Phys. Rev. E* 62, 1290–1312.
- Braun, M., Hertrich, M. & Yaramanci, U. (2005). Study on complex inversion of magnetic resonance sounding signals. *Near Surface Geophysics* 3, 155–163.
- Müller-Petke, M. & Yaramanci, U. (2010). QT inversion — comprehensive use of the complete surface NMR data set. *Geophysics* 75, WA199–WA209.
- Rice, S.O. (1944). Mathematical analysis of random noise. *Bell Syst. Tech. J.* 23, 282–332.
