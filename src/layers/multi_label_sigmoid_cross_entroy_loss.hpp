#ifndef CAFFE_MULTI_LABEL_SIGMOID_CROSS_ENCTROP_LOSS_LAYER_HPP_
#define CAFFE_MULTI_LABEL_SIGMOID_CROSS_ENCTROP_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "../layer.hpp"
#include "../proto/caffe.pb.h"

#include "./loss_layer.hpp"
#include "./sigmoid_layer.hpp"

namespace caffe {

/**
 * follow the formulae given in D.W. Li, 2015
 * "Multi Attribute Learning for Pedestrian Attribute Recognition in Surveillance Scenarios"
 * formulae (5)
 * we build a new message in caffe.proto to capture the hyper-prameter "weights" of each 
 * attribute 
 * At last, the formulae seems just a little bit different compared to the original sigmoid
 * cross entroy loss
 *
 * Author: Luo Heng
 *
 * Date: Nov. 12, 2017 in Chengdu. JINRIYUEDU BOOKSTORE
 */
class MultiLabelSigmoidCrossEntropyLossLayer : public LossLayer {
 public:
  explicit MultiLabelSigmoidCrossEntropyLossLayer(const LayerParameter& param)
      : LossLayer(param),
          sigmoid_layer_(new SigmoidLayer(param)),
          sigmoid_output_(new Blob()) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "MultiLabelSigmoidCrossEntropyLoss"; }

 protected:
  /// @copydoc SigmoidCrossEntropyLossLayer
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  /// Read the normalization mode parameter and compute the normalizer based
  /// on the blob size.  If normalization_mode is VALID, the count of valid
  /// outputs will be read from valid_count, unless it is -1 in which case
  /// all outputs are assumed to be valid.
  virtual real_t get_normalizer(
      LossParameter_NormalizationMode normalization_mode, int valid_count);

  /// The internal SigmoidLayer used to map predictions to probabilities.
  shared_ptr<SigmoidLayer> sigmoid_layer_;
  /// sigmoid_output stores the output of the SigmoidLayer.
  shared_ptr<Blob> sigmoid_output_;
  /// bottom vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob*> sigmoid_bottom_vec_;
  /// top vector holder to call the underlying SigmoidLayer::Forward
  vector<Blob*> sigmoid_top_vec_;


	// don't support ignore label in this version
	// maybe in the next version

  /// Whether to ignore instances with a certain label.
//  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
//  int ignore_label_;
  /// How to normalize the loss.
  LossParameter_NormalizationMode normalization_;
  real_t normalizer_;
  int outer_num_, inner_num_;
	MultiLabelSigmoidCrossEntropyLossParameter MLSCELPara_;

	std::vector<real_t> positive_ratio_;
	// the pairs store the weights for POSITIVE label and NEGATIVE label, respectively
	std::vector<std::pair<real_t, real_t>  > attribute_weights_;
	real_t epsilon_;
	int attributes_number_;
	bool has_positive_ratio_;
};

}  // namespace caffe

#endif  // CAFFE_SIGMOID_CROSS_ENTROPY_LOSS_LAYER_HPP_
