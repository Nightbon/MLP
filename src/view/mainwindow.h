#pragma once
#include <QChart>
#include <QChartView>
#include <QLineSeries>
#include <QMainWindow>

#include "../controller/controller.h"

namespace Ui {
class MainWindow;
}

namespace s21 {

class MainWindow : public QMainWindow {
  Q_OBJECT

 public:
  explicit MainWindow(QWidget* parent = nullptr);
  ~MainWindow();

 private slots:
  void on_comboModelType_currentIndexChanged(int index);
  void on_spinLayers_valueChanged(int arg1);
  void on_btnLoadWeights_clicked();
  void on_btnSaveWeights_clicked();
  void on_btnClearCanvas_clicked();
  void on_btnClassify_clicked();
  void on_btnLoadBmp_clicked();
  void on_btnStartTrain_clicked();
  void on_btnStartKFold_clicked();
  void on_btnRunExperiment_clicked();

  void handleEpochCompleted(int epoch, double train_loss, double test_loss);
  void handleTrainingFinished(bool success);
  void handleKFoldGroupCompleted(int group, double loss);
  void handleKFoldFinished(bool success);
  void handleTestFinished(s21::Metrics m, bool success);

 private:
  Ui::MainWindow* ui;
  Controller controller_;
  QLineSeries* error_series_;

  void SetupGraph();
  void ResetChart(const QString& title);
  void SetUIEnabled(bool enabled);
};

}  // namespace s21