#include "MediumFactory.hpp"

#include "HomogeneousMedium.hpp"
#include "AtmosphericMedium.hpp"
#include "ExponentialMedium.hpp"
#include "VoxelMedium.hpp"
#include "FunctionSpaceGaussianProcessMedium.hpp"
#include "WeightSpaceGaussianProcessMedium.hpp"
#include "SparseConvolutionNoiseMedium.hpp"
#include "RenewalPlusFpGradientMedium.hpp"
#include "ClassicGlobalMedium.hpp"
#include "ClassicGlobalFpGradientMedium.hpp"
#include "FirstHitGpisMedium.hpp"
#include "ClassicLocalMedium.hpp"
#include "GlobalConditionalMedium.hpp"
#include "SparseHistoryGpisMedium.hpp"
#include "SingleHistoryGpisMedium.hpp"
#include "PoissonEllipsoidMedium.hpp"
#ifdef TUNGSTEN_ENABLE_NEURAL_RENEWAL
#include "NeuralRenewalMedium.hpp"
#define NEURAL_RENEWAL_FACTORY_ENTRY {"neural_causal_renewal", std::make_shared<NeuralRenewalMedium>},
#else
#define NEURAL_RENEWAL_FACTORY_ENTRY
#endif

namespace Tungsten {

DEFINE_STRINGABLE_ENUM(MediumFactory, "medium", ({
    {"homogeneous", std::make_shared<HomogeneousMedium>},
    {"atmosphere", std::make_shared<AtmosphericMedium>},
    {"exponential", std::make_shared<ExponentialMedium>},
    {"voxel", std::make_shared<VoxelMedium>},
    {"gaussian_process", std::make_shared<FunctionSpaceGaussianProcessMedium>},                 // Backwards compatability with old files
    {"function_space_gaussian_process", std::make_shared<FunctionSpaceGaussianProcessMedium>}, 
    {"weight_space_gaussian_process", std::make_shared<WeightSpaceGaussianProcessMedium>},
    {"sparse_conv_noise", std::make_shared<SparseConvolutionNoiseMedium>},
    {"renewal_plus_fp_gradient", std::make_shared<RenewalPlusFpGradientMedium>},
    {"classic_global", std::make_shared<ClassicGlobalMedium>},
    {"classic_global_fp_gradient", std::make_shared<ClassicGlobalFpGradientMedium>},
    {"first_hit_gpis", std::make_shared<FirstHitGpisMedium>},
    {"classic_local", std::make_shared<ClassicLocalMedium>},
    {"global_conditional", std::make_shared<GlobalConditionalMedium>},
    {"sparse_history_gpis", std::make_shared<SparseHistoryGpisMedium>},
    {"single_history_gpis", std::make_shared<SingleHistoryGpisMedium>},
    {"poisson_ellipsoid_local", std::make_shared<PoissonEllipsoidMedium>},
    NEURAL_RENEWAL_FACTORY_ENTRY
}))

#undef NEURAL_RENEWAL_FACTORY_ENTRY

}
