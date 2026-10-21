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
