#pragma once
#include <span>
#include <string>
#include <vector>

#include "perceptron_interface.h"

namespace s21 {
struct Node;

struct Connection {
  double weight;
  Node* target;
};

struct Node {
  double value = 0.0;
  double error = 0.0;
  double bias = 0.0;
  std::vector<Connection> connections;
};

class GraphModel : public IPerceptron {
 public:
  GraphModel(size_t hidden_layers, size_t units_per_layer);

  std::vector<double> Predict(std::span<const double> input) override;
  void TrainEpoch(std::vector<Sample>& data, double lr) override;

  void SaveWeights(const std::string& path) const override;
  void LoadWeights(const std::string& path) override;
  void SetLayers(size_t hidden_layers) override;

 private:
  std::vector<std::vector<Node>> layers_;
  size_t hidden_layers_count_;
  size_t units_per_layer_;

  void LinkNodes();
};
}  // namespace s21