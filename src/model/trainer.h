#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "data_reader.h"
#include "graph_model.h"
#include "matrix_model.h"
#include "perceptron_interface.h"

namespace s21 {

struct EpochResult {
  int epoch = 0;
  double train_loss = 0.0;
  double test_loss = 0.0;
};

struct KFoldResult {
  int fold = 0;
  double loss = 0.0;
};

class Trainer {
 public:
  using ProgressCallback = std::function<void(int, double, double)>;
  bool Train(IPerceptron& model, const std::string& train_path,
             const std::string& test_path, int epochs,
             ProgressCallback callback, double learning_rate = 0.05) {
    auto train_data = DatasetReader::LoadDataset(train_path);

    auto test_data = DatasetReader::LoadDataset(test_path);

    if (train_data.empty() || test_data.empty()) {
      return false;
    }

    for (int e = 1; e <= epochs; ++e) {
      model.TrainEpoch(train_data, learning_rate);

      double train_loss = model.CalculateCrossEntropy(train_data);

      double test_loss = model.CalculateCrossEntropy(test_data);

      callback(e, train_loss, test_loss);
    }

    return true;
  }

  using KFoldCallback = std::function<void(int, double)>;

  bool RunKFold(bool is_graph, int layers, const std::string& path,
                int k_groups, KFoldCallback callback,
                double learning_rate = 0.01) {
    auto data = DatasetReader::LoadDataset(path);

    size_t k = std::max<size_t>(2, static_cast<size_t>(k_groups));

    auto splits = DatasetReader::BuildKFold(data, k);

    if (splits.empty()) {
      return false;
    }

    for (size_t i = 0; i < splits.size(); ++i) {
      std::unique_ptr<IPerceptron> model;

      if (is_graph) {
        model = std::make_unique<GraphModel>(layers, 128);

      } else {
        model = std::make_unique<MatrixModel>(layers, 128);
      }

      auto& split = splits[i];

      model->TrainEpoch(split.train, learning_rate);

      double loss = model->CalculateCrossEntropy(split.test);

      callback(static_cast<int>(i + 1), loss);
    }

    return true;
  }

 private:
  std::unique_ptr<IPerceptron> CreateModel(bool is_graph, int layers) {
    if (is_graph) {
      return std::make_unique<GraphModel>(layers, 128);
    }

    return std::make_unique<MatrixModel>(layers, 128);
  }
};

}  // namespace s21