#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include "metrics_collector.h"
#include "model_types.h"

namespace s21 {
class IPerceptron {
 public:
  virtual ~IPerceptron() = default;
  virtual std::vector<double> Predict(std::span<const double> input) = 0;
  virtual void TrainEpoch(std::vector<Sample>& data, double lr) = 0;
  virtual void SaveWeights(const std::string& path) const = 0;
  virtual void LoadWeights(const std::string& path) = 0;
  virtual void SetLayers(size_t hidden_layers) = 0;

  Metrics RunTest(const std::vector<Sample>& data, double part) {
    if (data.empty()) return {};
    std::vector<int> predictions;
    std::vector<int> targets;
    part = std::clamp(part, 0.0, 1.0);
    size_t count = static_cast<size_t>(data.size() * part);
    if (count == 0) count = 1;
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < count; ++i) {
      auto probs = Predict(data[i].data);
      int res = static_cast<int>(std::distance(
          probs.begin(), std::max_element(probs.begin(), probs.end())));
      predictions.push_back(res);
      targets.push_back(data[i].label);
    }
    auto end = std::chrono::high_resolution_clock::now();
    Metrics m = MetricsCollector::Calculate(predictions, targets);
    m.time = std::chrono::duration<double>(end - start).count();
    return m;
  }

  double CalculateCrossEntropy(const std::vector<Sample>& data) {
    if (data.empty()) {
      return 0.0;
    }

    double total_loss = 0.0;

    for (const auto& sample : data) {
      auto probs = Predict(sample.data);

      double p = std::max(probs[sample.label], 1e-15);

      total_loss -= std::log(p);
    }

    return total_loss / static_cast<double>(data.size());
  }

 protected:
  void Softmax(std::vector<double>& v) const {
    double max_val = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (auto& x : v) {
      x = std::exp(x - max_val);
      sum += x;
    }
    for (auto& x : v) x /= sum;
  }
  double Sigmoid(double x) const noexcept { return 1.0 / (1.0 + std::exp(-x)); }
  double SigmoidDerivative(double s) const noexcept { return s * (1.0 - s); }
};
}  // namespace s21