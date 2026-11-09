The Gaussian, mean-field and straight-flight code in this directory was ported
from the sibling `macrofacet` project (`include/macrofacet`, `src`) on 2026-10-20.
It is compiled locally: neither a sibling checkout nor a baked density field is
needed at runtime. Namespace and include paths are isolated; Random can consume
Tungsten's path sampler. The stable SE birth envelope and flux-gradient sampling
retain the source algorithm. Segment extensions are implemented separately.

Tungsten's `global_conditional` keeps only the latest full observation. It
renews that observation at GP collisions and geometry endpoints, using the
positive-endpoint sampler at ordinary surfaces. Tracking subdivisions do not
renew it. This is not full conditioning on the path's survival.

The medium option `non_gp_surface_policy` controls ordinary non-forward
geometry events. The default `retain_endpoint` keeps the existing observation
and consumes no additional random numbers. `restart_exterior` replaces it with
a fresh point-prior value and gradient conditioned on `F > 0`, independent of
the previous observation. The path tracer applies this once before both direct
light connections and BSDF continuation, including its resumable path. Forward
boundaries keep the existing endpoint behavior; actual medium changes still
reset the context. This is a comparison of memory approximations, not full
conditioning on the incoming segment's survival or exact Global transport.
