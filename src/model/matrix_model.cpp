#include "matrix_model.h"

#include <fstream>
#include <random>

namespace s21 {

MatrixModel::MatrixModel(size_t hidden_layers, size_t units_per_layer)
    : hidden_layers_count_(hidden_layers), units_per_layer_(units_per_layer) {
  SetLayers(hidden_layers);
}

void MatrixModel::ReallocateBuffers() {
  if (layers_.empty()) return;
  activations_.assign(layers_.size() + 1, {});
  activations_[0].assign(kInputSize, 0.0);
  for (size_t i = 0; i < layers_.size(); ++i)
    activations_[i + 1].assign(layers_[i].cols, 0.0);

  size_t max_nodes = std::max({kInputSize, units_per_layer_, kOutputSize});
  errors_.assign(max_nodes, 0.0);
  gradients_.assign(max_nodes, 0.0);
  next_errors_.assign(max_nodes, 0.0);
}

void MatrixModel::SetLayers(size_t hidden_layers) {
  hidden_layers_count_ = hidden_layers;
  layers_.clear();
  layers_.push_back({kInputSize, units_per_layer_,
                     std::vector<double>(kInputSize * units_per_layer_),
                     std::vector<double>(units_per_layer_)});
  for (size_t i = 0; i < hidden_layers_count_ - 1; ++i)
    layers_.push_back({units_per_layer_, units_per_layer_,
                       std::vector<double>(units_per_layer_ * units_per_layer_),
                       std::vector<double>(units_per_layer_)});
  layers_.push_back({units_per_layer_, kOutputSize,
                     std::vector<double>(units_per_layer_ * kOutputSize),
                     std::vector<double>(kOutputSize)});

  ReallocateBuffers();
  std::mt19937 gen(std::random_device{}());
  for (auto& l : layers_) {
    double lim = std::sqrt(6.0 / (double)(l.rows + l.cols));
    std::uniform_real_distribution<double> dist(-lim, lim);
    for (auto& w : l.weights) w = dist(gen);
  }
}

std::vector<double> MatrixModel::Predict(std::span<const double> input) {
  if (input.size() != kInputSize) {
    throw std::runtime_error("Invalid input size");
  }
  std::copy(input.begin(), input.end(), activations_[0].begin());
  for (size_t l = 0; l < layers_.size(); ++l) {
    const auto& layer = layers_[l];
    auto& curr = activations_[l + 1];
    std::fill(curr.begin(), curr.end(), 0.0);
    for (size_t i = 0; i < layer.rows; ++i) {
      double val = activations_[l][i];
      for (size_t j = 0; j < layer.cols; ++j)
        curr[j] += val * layer.weights[i * layer.cols + j];
    }
    for (size_t j = 0; j < layer.cols; ++j) curr[j] += layer.biases[j];

    if (l < layers_.size() - 1) {
      for (auto& x : curr) x = Sigmoid(x);
    } else {
      Softmax(curr);
    }
  }
  return activations_.back();
}

void MatrixModel::TrainEpoch(std::vector<Sample>& data, double lr) {
  static std::mt19937 gen(std::random_device{}());
  std::shuffle(data.begin(), data.end(), gen);
  for (const auto& s : data) {
    Predict(s.data);
    for (size_t i = 0; i < kOutputSize; ++i)
      errors_[i] = activations_.back()[i] - (size_t(s.label) == i ? 1.0 : 0.0);
    for (int l = (int)layers_.size() - 1; l >= 0; --l) {
      auto& layer = layers_[l];
      auto& curr_act = activations_[l + 1];
      for (size_t j = 0; j < layer.cols; ++j)
        gradients_[j] = (l == (int)layers_.size() - 1)
                            ? errors_[j]
                            : errors_[j] * SigmoidDerivative(curr_act[j]);
      std::fill(next_errors_.begin(), next_errors_.begin() + layer.rows, 0.0);
      for (size_t i = 0; i < layer.rows; ++i) {
        for (size_t j = 0; j < layer.cols; ++j) {
          next_errors_[i] += layer.weights[i * layer.cols + j] * gradients_[j];
          layer.weights[i * layer.cols + j] -=
              lr * activations_[l][i] * gradients_[j];
        }
      }
      for (size_t j = 0; j < layer.cols; ++j) {
        layer.biases[j] -= lr * gradients_[j];
        errors_[j] = next_errors_[j];
      }
      for (size_t i = 0; i < layer.rows; ++i) errors_[i] = next_errors_[i];
    }
  }
}

void MatrixModel::SaveWeights(const std::string& p) const {
  std::ofstream f(p, std::ios::binary);
  size_t s = layers_.size();
  f.write((char*)&s, sizeof(s));
  for (auto& l : layers_) {
    f.write((char*)&l.rows, sizeof(l.rows));
    f.write((char*)&l.cols, sizeof(l.cols));
    f.write((char*)l.weights.data(), l.weights.size() * sizeof(double));
    f.write((char*)l.biases.data(), l.biases.size() * sizeof(double));
  }
}

void MatrixModel::LoadWeights(const std::string& p) {
  std::ifstream f(p, std::ios::binary);

  if (!f.is_open()) throw std::runtime_error("File not found: " + p);

  size_t s;

  if (!f.read(reinterpret_cast<char*>(&s), sizeof(s))) return;

  layers_.clear();
  layers_.resize(s);
  for (auto& l : layers_) {
    f.read(reinterpret_cast<char*>(&l.rows), sizeof(l.rows));
    f.read(reinterpret_cast<char*>(&l.cols), sizeof(l.cols));
    l.weights.assign(l.rows * l.cols, 0.0);
    l.biases.assign(l.cols, 0.0);
    f.read(reinterpret_cast<char*>(l.weights.data()),
           l.weights.size() * sizeof(double));
    f.read(reinterpret_cast<char*>(l.biases.data()),
           l.biases.size() * sizeof(double));
  }
  hidden_layers_count_ = static_cast<int>(s) - 1;
  units_per_layer_ = layers_[0].cols;
  ReallocateBuffers();
}

}  // namespace s21