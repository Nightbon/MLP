#include "controller.h"

#include <QtConcurrent>

#include "../model/data_reader.h"
#include "../model/graph_model.h"
#include "../model/matrix_model.h"

namespace s21 {

Controller::Controller(QObject* parent) : QObject(parent) {
  SwitchModel(false, 2);
}

void Controller::SwitchModel(bool isGraph, int layers) {
  is_graph_ = isGraph;
  current_layers_ = layers;
  if (isGraph)
    model_ = std::make_unique<GraphModel>(layers, 128);
  else
    model_ = std::make_unique<MatrixModel>(layers, 128);
}

std::vector<double> Controller::Predict(std::span<const double> input) {
  return model_->Predict(input);
}
void Controller::SaveWeights(const std::string& path) {
  model_->SaveWeights(path);
}
void Controller::LoadWeights(const std::string& path) {
  model_->LoadWeights(path);
}

void Controller::StartTraining(const std::string& train_p,
                               const std::string& test_p, int epochs) {
  (void)QtConcurrent::run(&Controller::BackgroundTrain, this, train_p, test_p,
                          epochs);
}

void Controller::StartKFold(const std::string& path, int k_groups) {
  (void)QtConcurrent::run(&Controller::BackgroundKFold, this, path, k_groups);
}

void Controller::StartTest(const std::string& path, double part) {
  (void)QtConcurrent::run(&Controller::BackgroundTest, this, path, part);
}

void Controller::BackgroundTrain(std::string train_path, std::string test_path,
                                 int epochs) {
  bool success =
      trainer_.Train(*model_, train_path, test_path, epochs,

                     [this](int epoch, double train_loss, double test_loss) {
                       emit epochCompleted(epoch, train_loss, test_loss);
                     });

  emit trainingFinished(success);
}

void Controller::BackgroundKFold(std::string path, int k_groups) {
  bool success = trainer_.RunKFold(
      is_graph_, current_layers_, path, k_groups,

      [this](int fold, double loss) { emit kFoldGroupCompleted(fold, loss); });

  emit kFoldFinished(success);
}

void Controller::BackgroundTest(std::string path, double part) {
  auto test_data = DatasetReader::LoadDataset(path);
  if (test_data.empty()) {
    emit testFinished(Metrics{}, false);
  } else {
    emit testFinished(model_->RunTest(test_data, part), true);
  }
}

}  // namespace s21