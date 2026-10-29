#pragma once
#include <QImage>
#include <QMouseEvent>
#include <QPainter>
#include <QResizeEvent>
#include <QWidget>
#include <algorithm>
#include <vector>

namespace s21 {

class PaintWidget : public QWidget {
  Q_OBJECT
 public:
  explicit PaintWidget(QWidget* parent = nullptr) : QWidget(parent) {
    image_ = QImage(280, 280, QImage::Format_Grayscale8);
    image_.fill(Qt::black);
  }

  void clear() {
    image_.fill(Qt::black);
    update();
  }

  std::vector<double> getScaledData() {
    constexpr int kTargetDim = 28;
    constexpr int kLetterDim = 20;

    std::vector<double> data(kTargetDim * kTargetDim, 0.0);
    int min_x = image_.width(), min_y = image_.height();
    int max_x = 0, max_y = 0;
    bool found = false;

    for (int y = 0; y < image_.height(); ++y) {
      const uchar* row = image_.constScanLine(y);
      for (int x = 0; x < image_.width(); ++x) {
        if (row[x] > 20) {  // 20 — фильтр минимальной яркости/грязи
          min_x = std::min(min_x, x);
          max_x = std::max(max_x, x);
          min_y = std::min(min_y, y);
          max_y = std::max(max_y, y);
          found = true;
        }
      }
    }

    if (!found) return data;

    // Обрезка до границы букв + масштабирование до 20x20
    QImage cropped =
        image_.copy(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
    QImage scaled = cropped.scaled(kLetterDim, kLetterDim, Qt::KeepAspectRatio,
                                   Qt::SmoothTransformation);

    // Центрирование рисунка внутри 28x28 окна (По канонам формата EMNIST)
    QImage final_img(kTargetDim, kTargetDim, QImage::Format_Grayscale8);
    final_img.fill(Qt::black);

    QPainter p(&final_img);
    int offset_x = (kTargetDim - scaled.width()) / 2;
    int offset_y = (kTargetDim - scaled.height()) / 2;
    p.drawImage(offset_x, offset_y, scaled);
    p.end();

    // Сборка массива, аналогично читаем напрямую
    for (int y = 0; y < kTargetDim; ++y) {
      const uchar* row = final_img.constScanLine(y);
      for (int x = 0; x < kTargetDim; ++x) {
        data[static_cast<size_t>(y * kTargetDim + x)] = row[x] / 255.0;
      }
    }
    return data;
  }

 protected:
  void resizeEvent(QResizeEvent* event) override {
    if (image_.size() == event->size()) return;

    QImage new_image(event->size(), QImage::Format_Grayscale8);
    new_image.fill(Qt::black);

    // Перерисовываем то, что было, на новый увеличенный/уменьшенный канвас
    QPainter p(&new_image);
    p.drawImage(0, 0, image_);

    image_ = new_image;
  }

  void mouseMoveEvent(QMouseEvent* event) override {
    if (event->buttons() & Qt::LeftButton) {
      QPainter painter(&image_);
      painter.setPen(
          QPen(Qt::white, 25, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
      painter.drawLine(last_point_, event->pos());
      last_point_ = event->pos();
      update();  // запрос перерисовки (отрабатывает paintEvent)
    }
  }

  void mousePressEvent(QMouseEvent* event) override {
    if (event->button() == Qt::LeftButton) {
      last_point_ = event->pos();
    }
  }

  void paintEvent(QPaintEvent*) override {
    QPainter painter(this);
    painter.drawImage(0, 0, image_);
  }

 private:
  QImage image_;
  QPoint last_point_;
};

}  // namespace s21