#include "MediumFactory.hpp"

#include "HomogeneousMedium.hpp"
#include "AtmosphericMedium.hpp"
#include "ExponentialMedium.hpp"
#include "VoxelMedium.hpp"
#include "FunctionSpaceGaussianProcessMedium.hpp"
#include "WeightSpaceGaussianProcessMedium.hpp"
#include "SparseConvolutionNoiseMedium.hpp"
#include "ClassicGlobalMedium.hpp"
#include "ClassicLocalMedium.hpp"
#include "GlobalConditionalMedium.hpp"

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
    {"classic_global", std::make_shared<ClassicGlobalMedium>},
    {"classic_local", std::make_shared<ClassicLocalMedium>},
    {"global_conditional", std::make_shared<GlobalConditionalMedium>},
}))

}
