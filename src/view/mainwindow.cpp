#include "mainwindow.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QPainter>
#include <QValueAxis>

#include "ui_mainwindow.h"

namespace s21 {

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent), ui(new Ui::MainWindow), error_series_(nullptr) {
  ui->setupUi(this);
  SetupGraph();

  QFont font = ui->lblLetterResult->font();
  font.setPointSize(64);
  font.setBold(true);
  ui->lblLetterResult->setFont(font);
  ui->lblLetterResult->setStyleSheet(
      "color: #2F3640; background: transparent;");
  ui->lblLetterResult->setText("?");
  ui->txtTrainingLog->setFocusPolicy(Qt::NoFocus);

  connect(&controller_, &Controller::epochCompleted, this,
          &MainWindow::handleEpochCompleted);
  connect(&controller_, &Controller::trainingFinished, this,
          &MainWindow::handleTrainingFinished);
  connect(&controller_, &Controller::kFoldGroupCompleted, this,
          &MainWindow::handleKFoldGroupCompleted);
  connect(&controller_, &Controller::kFoldFinished, this,
          &MainWindow::handleKFoldFinished);
  connect(&controller_, &Controller::testFinished, this,
          &MainWindow::handleTestFinished);
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::SetupGraph() {
  error_series_ = new QLineSeries();
  error_series_->setPointsVisible(true);
  error_series_->setPointLabelsVisible(false);
  error_series_->setPointLabelsFormat("@yPoint");

  auto* chart = new QChart();
  chart->addSeries(error_series_);
  chart->createDefaultAxes();
  chart->setTitle("Progress (Cross-Entropy Loss)");
  chart->legend()->hide();

  auto* axisX = qobject_cast<QValueAxis*>(chart->axes(Qt::Horizontal).first());
  if (axisX) {
    axisX->setLabelFormat("%d");
    axisX->setTitleText("Step");
  }

  auto* axisY = qobject_cast<QValueAxis*>(chart->axes(Qt::Vertical).first());
  if (axisY) {
    axisY->setLabelFormat("%.4f");
    axisY->setTitleText("Loss");
  }

  ui->chartView->setChart(chart);
  ui->chartView->setRenderHint(QPainter::Antialiasing);
}

void MainWindow::ResetChart(const QString& title) {
  auto* old_chart = ui->chartView->chart();
  auto* new_chart = new QChart();
  new_chart->setTitle(title);
  new_chart->legend()->hide();

  if (old_chart) old_chart->removeSeries(error_series_);

  error_series_->setPointLabelsVisible(true);
  error_series_->setPointLabelsFormat("@yPoint");

  new_chart->addSeries(error_series_);
  new_chart->createDefaultAxes();

  auto* axisX =
      qobject_cast<QValueAxis*>(new_chart->axes(Qt::Horizontal).first());
  if (axisX) {
    axisX->setLabelFormat("%d");
    axisX->setTitleText("Step");
    int count = error_series_->count();
    if (count > 1) axisX->setTickCount(count);
  }

  auto* axisY =
      qobject_cast<QValueAxis*>(new_chart->axes(Qt::Vertical).first());
  if (axisY) {
    axisY->setLabelFormat("%.4f");
    axisY->setTitleText("Loss");
  }

  ui->chartView->setChart(new_chart);
}

void MainWindow::SetUIEnabled(bool enabled) {
  ui->groupMain->setEnabled(enabled);
  ui->groupExperiment->setEnabled(enabled);
  ui->spinEpochs->setEnabled(enabled);
  ui->spinKFold->setEnabled(enabled);
  ui->btnStartTrain->setEnabled(enabled);
  ui->btnStartKFold->setEnabled(enabled);
  ui->txtTrainingLog->setEnabled(true);
}

void MainWindow::on_comboModelType_currentIndexChanged(int index) {
  controller_.SwitchModel(index == 1, ui->spinLayers->value());
}
void MainWindow::on_spinLayers_valueChanged(int val) {
  controller_.SwitchModel(ui->comboModelType->currentIndex() == 1, val);
}

void MainWindow::on_btnLoadWeights_clicked() {
  QString path = QFileDialog::getOpenFileName(this, "Open", "", "*.bin");
  if (path.isEmpty()) return;
  try {
    controller_.LoadWeights(path.toStdString());
    QMessageBox::information(this, "Success", "Weights loaded!");
  } catch (const std::exception& e) {
    QMessageBox::critical(this, "Error", e.what());
  }
}

void MainWindow::on_btnSaveWeights_clicked() {
  QString path = QFileDialog::getSaveFileName(this, "Save", "", "*.bin");
  if (path.isEmpty()) return;
  if (!path.endsWith(".bin")) path += ".bin";
  try {
    controller_.SaveWeights(path.toStdString());
    QMessageBox::information(this, "Success", "Weights saved!");
  } catch (const std::exception& e) {
    QMessageBox::critical(this, "Error", e.what());
  }
}

void MainWindow::on_btnClearCanvas_clicked() {
  ui->paintWidget->clear();
  ui->lblLetterResult->setText("?");
}

void MainWindow::on_btnClassify_clicked() {
  std::vector<double> data = ui->paintWidget->getScaledData();
  if (data.empty()) return;
  std::vector<double> probs = controller_.Predict(data);
  int res = static_cast<int>(std::distance(
      probs.begin(), std::max_element(probs.begin(), probs.end())));
  ui->lblLetterResult->setText(QString("%1").arg(static_cast<char>('A' + res)));
}

void MainWindow::on_btnLoadBmp_clicked() {
  QString path = QFileDialog::getOpenFileName(this, "Open BMP", "", "*.bmp");
  if (path.isEmpty()) return;
  QImage img(path);
  if (img.isNull()) return;
  img = img.convertToFormat(QImage::Format_Grayscale8);
  if (qGray(img.pixel(0, 0)) > 127) img.invertPixels();
  int min_x = img.width(), min_y = img.height(), max_x = 0, max_y = 0;
  bool found = false;
  for (int y = 0; y < img.height(); ++y) {
    const uchar* row = img.constScanLine(y);
    for (int x = 0; x < img.width(); ++x)
      if (row[x] > 20) {
        min_x = std::min(min_x, x);
        min_y = std::min(min_y, y);
        max_x = std::max(max_x, x);
        max_y = std::max(max_y, y);
        found = true;
      }
  }
  if (!found) return;
  QImage cropped = img.copy(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
  QImage scaled =
      cropped.scaled(20, 20, Qt::KeepAspectRatio, Qt::SmoothTransformation);
  QImage final_img(28, 28, QImage::Format_Grayscale8);
  final_img.fill(Qt::black);
  QPainter painter(&final_img);
  painter.drawImage((28 - scaled.width()) / 2, (28 - scaled.height()) / 2,
                    scaled);
  painter.end();
  std::vector<double> input(784);
  for (int y = 0; y < 28; ++y) {
    const uchar* row = final_img.constScanLine(y);
    for (int x = 0; x < 28; ++x)
      input[static_cast<size_t>(y * 28 + x)] = row[x] / 255.0;
  }
  auto probs = controller_.Predict(input);
  int res = static_cast<int>(std::distance(
      probs.begin(), std::max_element(probs.begin(), probs.end())));
  ui->lblLetterResult->setText(QString("%1").arg(static_cast<char>('A' + res)));
}

void MainWindow::on_btnStartTrain_clicked() {
  QString train_p =
      QFileDialog::getOpenFileName(this, "Train CSV", "", "*.csv");

  QString test_p = QFileDialog::getOpenFileName(this, "Test CSV", "", "*.csv");

  if (train_p.isEmpty() || test_p.isEmpty()) {
    return;
  }

  error_series_->clear();

  ui->txtTrainingLog->clear();

  SetUIEnabled(false);

  ui->lblTrainStatus->setText("Epoch 1 training...");

  controller_.StartTraining(train_p.toStdString(), test_p.toStdString(),
                            ui->spinEpochs->value());
}

void MainWindow::handleEpochCompleted(int epoch, double train_loss,
                                      double test_loss) {
  error_series_->append(static_cast<double>(epoch), test_loss);

  auto* chart = ui->chartView->chart();

  if (chart) {
    auto* axisX =
        qobject_cast<QValueAxis*>(chart->axes(Qt::Horizontal).first());

    auto* axisY = qobject_cast<QValueAxis*>(chart->axes(Qt::Vertical).first());

    if (axisX) {
      axisX->setRange(1, epoch);
    }

    if (axisY) {
      double min_y = test_loss * 0.9;

      double max_y = test_loss * 1.1;

      for (const QPointF& p : error_series_->points()) {
        min_y = std::min(min_y, p.y());

        max_y = std::max(max_y, p.y());
      }

      axisY->setRange(min_y * 0.95, max_y * 1.05);
    }
  }

  ui->lblTrainStatus->setText(QString("Epoch %1 training...").arg(epoch + 1));

  ui->txtTrainingLog->appendPlainText(
      QString("Epoch %1 | Train Loss: %2 | Test Loss: %3")
          .arg(epoch)
          .arg(train_loss, 0, 'f', 4)
          .arg(test_loss, 0, 'f', 4));
}

void MainWindow::handleTrainingFinished(bool success) {
  SetUIEnabled(true);

  ui->lblTrainStatus->setText(success ? "Training Complete!"
                                      : "Training Failed!");
}

void MainWindow::on_btnStartKFold_clicked() {
  QString path = QFileDialog::getOpenFileName(this, "KFold CSV", "", "*.csv");

  if (path.isEmpty()) {
    return;
  }

  error_series_->clear();

  ui->txtTrainingLog->clear();

  auto* chart = ui->chartView->chart();

  if (chart) {
    auto axesX = chart->axes(Qt::Horizontal);

    auto axesY = chart->axes(Qt::Vertical);

    if (!axesX.isEmpty()) {
      auto* axisX = qobject_cast<QValueAxis*>(axesX.first());

      if (axisX) {
        axisX->setRange(1, 2);
      }
    }

    if (!axesY.isEmpty()) {
      auto* axisY = qobject_cast<QValueAxis*>(axesY.first());

      if (axisY) {
        axisY->setRange(0, 1);
      }
    }
  }

  SetUIEnabled(false);

  ui->lblTrainStatus->setText("Fold 1 running...");

  controller_.StartKFold(path.toStdString(), ui->spinKFold->value());
}

void MainWindow::handleKFoldGroupCompleted(int group, double loss) {
  error_series_->append(static_cast<double>(group), loss);

  auto* chart = ui->chartView->chart();

  if (chart) {
    auto axesX = chart->axes(Qt::Horizontal);

    auto axesY = chart->axes(Qt::Vertical);

    if (!axesX.isEmpty()) {
      auto* axisX = qobject_cast<QValueAxis*>(axesX.first());

      if (axisX) {
        axisX->setRange(1, std::max(2, group));
      }
    }

    if (!axesY.isEmpty()) {
      auto* axisY = qobject_cast<QValueAxis*>(axesY.first());

      if (axisY) {
        double min_loss = loss;

        double max_loss = loss;

        for (const QPointF& p : error_series_->points()) {
          min_loss = std::min(min_loss, p.y());

          max_loss = std::max(max_loss, p.y());
        }

        axisY->setRange(min_loss * 0.9, max_loss * 1.1);
      }
    }
  }

  ui->lblTrainStatus->setText(QString("Fold %1 running...").arg(group + 1));

  ui->txtTrainingLog->appendPlainText(
      QString("Fold %1 | Loss: %2").arg(group).arg(loss, 0, 'f', 4));
}

void MainWindow::handleKFoldFinished(bool success) {
  SetUIEnabled(true);

  ui->lblTrainStatus->setText(success ? "K-Fold Complete!" : "K-Fold Failed!");
}

void MainWindow::on_btnRunExperiment_clicked() {
  QString path = QFileDialog::getOpenFileName(this, "Test CSV", "", "*.csv");
  if (path.isEmpty()) return;
  SetUIEnabled(false);
  ui->lblAccuracy->setText("Testing...");
  controller_.StartTest(path.toStdString(), ui->spinTestPart->value());
}

void MainWindow::handleTestFinished(s21::Metrics m, bool success) {
  SetUIEnabled(true);
  if (!success) {
    QMessageBox::warning(this, "Error", "Test failed.");
    return;
  }
  ui->lblAccuracy->setText(
      QString("Accuracy: %1%").arg(m.accuracy * 100.0, 0, 'f', 2));
  ui->lblPrecision->setText(
      QString("Precision: %1%").arg(m.precision * 100.0, 0, 'f', 2));
  ui->lblRecall->setText(
      QString("Recall: %1%").arg(m.recall * 100.0, 0, 'f', 2));
  ui->lblFMeasure->setText(
      QString("F-measure: %1%").arg(m.f_measure * 100.0, 0, 'f', 2));
  ui->lblTestTime->setText(QString("Time: %1s").arg(m.time, 0, 'f', 2));
}

}  // namespace s21