#pragma once
#include <nlohmann/json.hpp>

inline nlohmann::json singlePointNeuralFixture(double sigma=.02,double ell=.2,double residual=0)
{
    using J=nlohmann::json;
    const J names={"z_x","z_u","z_v","rho_xu","rho_xv","rho_uv"};
    return {{"schema","tungsten.single_point_history_mlp.v2"},
        {"architecture",{{"input_features",6},{"output_features",1},{"hidden_layers",2},{"width",2},{"activation","relu"}}},
        {"feature_order",names},{"feature_mean",{0,0,0,0,0,0}},{"feature_scale",{1,1,1,1,1,1}},
        {"feature_transform","identity"},{"mlp_dtype","float32"},{"probability_transform_dtype","float64"},
        {"layers",J::array({{{"type","linear"},{"weights",{{0,0,0,0,0,0},{0,0,0,0,0,0}}},{"bias",{0,0}}},
            {{"type","relu"}},{{"type","linear"},{"weights",{{0,0},{0,0}}},{"bias",{0,0}}},
            {{"type","relu"}},{{"type","linear"},{"weights",{{0,0}}},{"bias",{residual}}}})},
        {"output",{{"quantity","log_hazard_correction"},{"raw_quantity","residual_f"}}},
        {"deployment",{{"stage","inference_only"},{"correction_cap",2},{"applied_to_training_loss",false},{"lower_cap",nullptr}}},
        {"sigma",sigma},{"length_scale",ell},
        {"source_manifest",{{"schema","tungsten.single_point_history.v1"},{"status","complete"},
            {"sigma",sigma},{"length_scale",ell},
            {"source",{{"domain",{{"minimum",{-1,-1,-1}},{"maximum",{1,1,1}}}}}},
            {"sampling",{{"minimum_age_over_length",.05},{"minimum_separation_over_length",.01},
                {"anchor_implementation","shared_geometry_pool_v2"}}}}},
        {"training_domain",{{"minimum_features",{-100,-100,-100,-1,-1,-1}},
            {"maximum_features",{100,100,100,1,1,1}}}}};
}
