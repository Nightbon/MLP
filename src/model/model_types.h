#pragma once
#include <vector>

namespace s21 {

constexpr size_t kInputSize = 784;
constexpr size_t kOutputSize = 26;

struct Sample {
  int label;
  std::vector<double> data;
};

struct Metrics {
  double accuracy = 0.0;
  double precision = 0.0;
  double recall = 0.0;
  double f_measure = 0.0;
  double time = 0.0;
};

}  // namespace s21