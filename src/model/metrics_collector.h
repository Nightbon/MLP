#pragma once
#include <set>
#include <vector>

#include "model_types.h"

namespace s21 {
class MetricsCollector {
 public:
  static Metrics Calculate(const std::vector<int>& predictions,
                           const std::vector<int>& targets) {
    size_t size = predictions.size();

    if (size == 0 || size != targets.size()) {
      return Metrics{};
    }

    std::vector<double> tp(kOutputSize, 0.0);
    std::vector<double> fp(kOutputSize, 0.0);
    std::vector<double> fn(kOutputSize, 0.0);

    std::set<int> present_labels;

    int correct = 0;

    for (size_t i = 0; i < size; ++i) {
      int pred = predictions[i];
      int target = targets[i];

      present_labels.insert(target);

      if (pred == target) {
        tp[target]++;
        correct++;
      } else {
        fp[pred]++;
        fn[target]++;
      }
    }

    double precision = 0.0;
    double recall = 0.0;

    for (int label : present_labels) {
      double precision_i = 0.0;
      double recall_i = 0.0;

      if (tp[label] + fp[label] > 0.0) {
        precision_i = tp[label] / (tp[label] + fp[label]);
      }

      if (tp[label] + fn[label] > 0.0) {
        recall_i = tp[label] / (tp[label] + fn[label]);
      }

      precision += precision_i;
      recall += recall_i;
    }

    precision /= present_labels.size();
    recall /= present_labels.size();

    double f1 = 0.0;

    if (precision + recall > 0.0) {
      f1 = 2.0 * precision * recall / (precision + recall);
    }

    return {static_cast<double>(correct) / size, precision, recall, f1, 0.0};
  }
};
}  // namespace s21