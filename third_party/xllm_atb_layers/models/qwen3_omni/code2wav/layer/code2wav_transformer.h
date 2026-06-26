#ifndef ATB_SPEED_MODELS_CODE2WAV_TRANSFORMER_H
#define ATB_SPEED_MODELS_CODE2WAV_TRANSFORMER_H

#include "atb/atb_infer.h"
#include "atb_speed/base/hosttensor_binder.h"
#include "nlohmann/json.hpp"
#include "models/qwen3_omni/code2wav/operation/code2wav_operations.h"

namespace atb_speed {
namespace qwen3_omni {

atb::Status Qwen3Omni_Code2wav_TransformerLayer(const Code2WavTransformerLayerParam& param,
                         atb::Operation** operation);

}  // namespace qwen
}  // namespace atb_speed
#endif
