#include <algorithm>
#include <vector>

#include "./loss_layer.hpp"

namespace caffe {

void LossLayer::LayerSetUp(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(real_t(1));
  }
}

void LossLayer::Reshape(
    const vector<Blob*>& bottom, const vector<Blob*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "The data and label should have the same first dimension.";
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

real_t LossLayer::GetNormalizer(
    const LossParameter_NormalizationMode normalization_mode,
    const int outer_num, const int inner_num, const int valid_count) {
  real_t normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = real_t(outer_num * inner_num);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = real_t(outer_num * inner_num);
      } else {
        normalizer = real_t(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = real_t(outer_num);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = real_t(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(real_t(1.0), normalizer);
}

}  // namespace caffe
