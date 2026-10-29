#include <QApplication>
#include <QMetaType>

#include "model/model_types.h"
#include "view/mainwindow.h"

int main(int argc, char* argv[]) {
  QApplication a(argc, argv);

  s21::MainWindow w;
  w.show();

  return a.exec();
}