#pragma once
#include <QObject>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "../model/model_types.h"
#include "../model/perceptron_interface.h"
#include "../model/trainer.h"

namespace s21 {

class Controller : public QObject {
  Q_OBJECT
 public:
  explicit Controller(QObject* parent = nullptr);

  void SwitchModel(bool isGraph, int layers);
  std::vector<double> Predict(std::span<const double> input);
  void SaveWeights(const std::string& path);
  void LoadWeights(const std::string& path);

 public slots:
  void StartTraining(const std::string& train_p, const std::string& test_p,
                     int epochs);
  void StartKFold(const std::string& path, int k_groups);
  void StartTest(const std::string& path, double part);

 signals:
  void epochCompleted(int epoch, double train_loss, double test_loss);
  void trainingFinished(bool success);
  void kFoldGroupCompleted(int group, double loss);
  void kFoldFinished(bool success);
  void testFinished(Metrics m, bool success);

 private:
  std::unique_ptr<IPerceptron> model_;
  bool is_graph_ = false;
  int current_layers_ = 2;
  Trainer trainer_;

  void BackgroundTrain(std::string train_path, std::string test_path,
                       int epochs);
  void BackgroundKFold(std::string path, int k_groups);
  void BackgroundTest(std::string path, double part);
};

}  // namespace s21