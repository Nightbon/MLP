#pragma once
#include <span>
#include <string>
#include <vector>

#include "perceptron_interface.h"

namespace s21 {
struct MatrixLayer {
  size_t rows;
  size_t cols;
  std::vector<double> weights;
  std::vector<double> biases;
};

class MatrixModel : public IPerceptron {
 public:
  MatrixModel(size_t hidden_layers, size_t units_per_layer);

  std::vector<double> Predict(std::span<const double> input) override;
  void TrainEpoch(std::vector<Sample>& data, double lr) override;

  void SaveWeights(const std::string& path) const override;
  void LoadWeights(const std::string& path) override;
  void SetLayers(size_t hidden_layers) override;

 private:
  size_t hidden_layers_count_;
  size_t units_per_layer_;
  std::vector<MatrixLayer> layers_;
  mutable std::vector<std::vector<double>> activations_;

  std::vector<double> errors_;
  std::vector<double> gradients_;
  std::vector<double> next_errors_;

  void InitWeights();
  void ReallocateBuffers();
};
}  // namespace s21