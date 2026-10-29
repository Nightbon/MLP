#include "graph_model.h"

#include <fstream>
#include <random>

namespace s21 {
GraphModel::GraphModel(size_t h, size_t u)
    : hidden_layers_count_(h), units_per_layer_(u) {
  SetLayers(h);
}

void GraphModel::LinkNodes() {
  for (size_t i = 0; i < layers_.size() - 1; ++i) {
    for (size_t r = 0; r < layers_[i].size(); ++r) {
      layers_[i][r].connections.resize(layers_[i + 1].size());
      for (size_t c = 0; c < layers_[i + 1].size(); ++c)
        layers_[i][r].connections[c].target = &layers_[i + 1][c];
    }
  }
}

void GraphModel::SetLayers(size_t h) {
  hidden_layers_count_ = h;
  layers_.clear();
  layers_.resize(h + 2);
  layers_[0].resize(kInputSize);
  for (size_t i = 1; i <= h; ++i) layers_[i].resize(units_per_layer_);
  layers_.back().resize(kOutputSize);
  LinkNodes();
  std::mt19937 gen(std::random_device{}());
  for (size_t i = 0; i < layers_.size() - 1; ++i) {
    double lim =
        std::sqrt(6.0 / (double)(layers_[i].size() + layers_[i + 1].size()));
    std::uniform_real_distribution<double> dist(-lim, lim);
    for (auto& n : layers_[i])
      for (auto& c : n.connections) c.weight = dist(gen);
  }
}

std::vector<double> GraphModel::Predict(std::span<const double> input) {
  if (input.size() != kInputSize) {
    throw std::runtime_error("Invalid input size");
  }
  for (size_t i = 0; i < input.size(); ++i) layers_[0][i].value = input[i];
  for (size_t i = 0; i < layers_.size() - 1; ++i) {
    for (auto& next_node : layers_[i + 1]) next_node.value = 0.0;
    for (auto& node : layers_[i])
      for (auto& c : node.connections) c.target->value += node.value * c.weight;
    if (i < layers_.size() - 2)
      for (auto& n : layers_[i + 1]) n.value = Sigmoid(n.value + n.bias);
  }
  std::vector<double> out(kOutputSize);
  for (size_t k = 0; k < kOutputSize; ++k)
    out[k] = layers_.back()[k].value + layers_.back()[k].bias;

  Softmax(out);  // ПРОСТО Softmax

  for (size_t k = 0; k < kOutputSize; ++k) layers_.back()[k].value = out[k];
  return out;
}

void GraphModel::TrainEpoch(std::vector<Sample>& data, double lr) {
  static std::mt19937 gen(std::random_device{}());
  std::shuffle(data.begin(), data.end(), gen);
  for (const auto& s : data) {
    Predict(s.data);
    for (size_t i = 0; i < kOutputSize; ++i)
      layers_.back()[i].error =
          layers_.back()[i].value - (size_t(s.label) == i ? 1.0 : 0.0);
    for (int i = (int)layers_.size() - 2; i > 0; --i) {
      for (auto& node : layers_[i]) {
        double err = 0.0;
        for (auto& c : node.connections) err += c.weight * c.target->error;
        node.error = err * SigmoidDerivative(node.value);
      }
    }
    for (size_t i = 0; i < layers_.size() - 1; ++i) {
      for (auto& node : layers_[i])
        for (auto& c : node.connections)
          c.weight -= lr * node.value * c.target->error;
      for (auto& next_node : layers_[i + 1])
        next_node.bias -= lr * next_node.error;
    }
  }
}

void GraphModel::SaveWeights(const std::string& p) const {
  std::ofstream f(p, std::ios::binary);
  size_t s = layers_.size() - 1;
  f.write((char*)&s, sizeof(s));
  for (size_t i = 0; i < s; ++i) {
    size_t r = layers_[i].size(), c = layers_[i + 1].size();
    f.write((char*)&r, sizeof(r));
    f.write((char*)&c, sizeof(c));
    for (auto& n : layers_[i])
      for (auto& con : n.connections)
        f.write((char*)&con.weight, sizeof(double));
    for (auto& n : layers_[i + 1]) f.write((char*)&n.bias, sizeof(double));
  }
}

void GraphModel::LoadWeights(const std::string& p) {
  std::ifstream f(p, std::ios::binary);

  if (!f.is_open()) throw std::runtime_error("File not found: " + p);

  size_t s;
  if (!f.read(reinterpret_cast<char*>(&s), sizeof(s))) return;

  layers_.clear();
  layers_.resize(s + 1);
  for (size_t i = 0; i < s; ++i) {
    size_t r;
    size_t c;

    f.read(reinterpret_cast<char*>(&r), sizeof(r));
    f.read(reinterpret_cast<char*>(&c), sizeof(c));

    if (i == 0) layers_[0].resize(r);

    layers_[i + 1].resize(c);
    for (size_t j = 0; j < r; ++j) {
      layers_[i][j].connections.resize(c);
      for (size_t k = 0; k < c; ++k) {
        f.read(reinterpret_cast<char*>(&layers_[i][j].connections[k].weight),
               sizeof(double));
      }
    }
    for (size_t j = 0; j < c; ++j) {
      f.read(reinterpret_cast<char*>(&layers_[i + 1][j].bias), sizeof(double));
    }
  }
  hidden_layers_count_ = static_cast<int>(s) - 1;
  units_per_layer_ = layers_[1].size();
  LinkNodes();
}
}  // namespace s21