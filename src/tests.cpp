#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <vector>

#include "model/graph_model.h"
#include "model/matrix_model.h"
#include "model/metrics_collector.h"
#include "model/model_types.h"

TEST(MetricsTest, PerfectMetrics) {
  std::vector<int> predictions;
  std::vector<int> targets;

  for (int i = 0; i < 26; ++i) {
    predictions.push_back(i);
    targets.push_back(i);
  }

  s21::Metrics m = s21::MetricsCollector::Calculate(predictions, targets);

  EXPECT_DOUBLE_EQ(m.accuracy, 1.0);
  EXPECT_GT(m.f_measure, 0.99);
}
TEST(MetricsTest, PartialMetrics) {
  std::vector<int> predictions = {0, 0, 1, 1};
  std::vector<int> targets = {0, 1, 1, 0};

  s21::Metrics m = s21::MetricsCollector::Calculate(predictions, targets);

  EXPECT_DOUBLE_EQ(m.accuracy, 0.5);
  EXPECT_TRUE(m.precision > 0.0);
  EXPECT_TRUE(m.recall > 0.0);
}

TEST(MetricsTest, SingleClassOnly) {
  std::vector<int> pred = {0, 0, 0, 0};

  std::vector<int> target = {0, 0, 0, 0};

  auto m = s21::MetricsCollector::Calculate(pred, target);

  EXPECT_DOUBLE_EQ(m.accuracy, 1.0);
}

TEST(MatrixModelTest, OutputStructureAndSoftmax) {
  s21::MatrixModel model(2, 128);
  std::vector<double> dummy_input(784, 0.5);

  auto output = model.Predict(dummy_input);

  EXPECT_EQ(output.size(), 26);

  double sum = 0.0;
  for (double p : output) {
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);

    sum += p;
  }
  EXPECT_NEAR(sum, 1.0, 1e-5);
}

TEST(GraphModelTest, OutputStructureAndSoftmax) {
  s21::GraphModel model(3, 64);
  std::vector<double> dummy_input(784, 0.2);

  auto output = model.Predict(dummy_input);

  EXPECT_EQ(output.size(), 26);

  double sum = 0.0;
  for (double p : output) {
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    sum += p;
  }
  EXPECT_NEAR(sum, 1.0, 1e-5);
}

TEST(SerializationTest, MatrixSaveLoadIntegrity) {
  std::vector<double> dummy_input(784, 0.7);
  std::string test_file = "test_matrix_weights.bin";

  s21::MatrixModel model_a(2, 128);
  auto output_a = model_a.Predict(dummy_input);
  model_a.SaveWeights(test_file);

  s21::MatrixModel model_b(4, 30);
  model_b.LoadWeights(test_file);

  auto output_b = model_b.Predict(dummy_input);

  for (size_t i = 0; i < 26; ++i) {
    EXPECT_NEAR(output_a[i], output_b[i], 1e-6);
  }

  std::remove(test_file.c_str());
}

TEST(SerializationTest, EmptyFileLoad) {
  std::ofstream out("empty.bin", std::ios::binary);

  out.close();

  s21::MatrixModel model(2, 64);

  EXPECT_NO_THROW(model.LoadWeights("empty.bin"));

  std::remove("empty.bin");
}

TEST(SerializationTest, CrossArchitectureCompatibility) {
  std::vector<double> dummy_input(784, 0.3);
  std::string test_file = "test_cross_weights.bin";

  s21::GraphModel graph_model(2, 128);
  auto output_graph = graph_model.Predict(dummy_input);
  graph_model.SaveWeights(test_file);

  s21::MatrixModel matrix_model(2, 128);
  matrix_model.LoadWeights(test_file);
  auto output_matrix = matrix_model.Predict(dummy_input);

  for (size_t i = 0; i < 26; ++i) {
    EXPECT_NEAR(output_graph[i], output_matrix[i], 1e-6);
  }

  std::remove(test_file.c_str());
}

TEST(MatrixModelTest, TrainingReducesError) {
  s21::MatrixModel model(2, 128);
  s21::Sample dummy_sample;
  dummy_sample.label = 2;
  dummy_sample.data = std::vector<double>(s21::kInputSize, 0.5);

  std::vector<s21::Sample> dataset = {dummy_sample};

  double loss_before = model.CalculateCrossEntropy(dataset);

  for (int i = 0; i < 5; ++i) {
    model.TrainEpoch(dataset, 0.5);
  }

  double loss_after = model.CalculateCrossEntropy(dataset);

  EXPECT_LT(loss_after, loss_before);
}

TEST(GraphModelTest, TrainingReducesError) {
  s21::GraphModel model(2, 128);
  s21::Sample dummy_sample;
  dummy_sample.label = 4;
  dummy_sample.data = std::vector<double>(s21::kInputSize, 0.8);
  std::vector<s21::Sample> dataset = {dummy_sample};

  double loss_before = model.CalculateCrossEntropy(dataset);
  for (int i = 0; i < 5; ++i) {
    model.TrainEpoch(dataset, 0.5);
  }
  double loss_after = model.CalculateCrossEntropy(dataset);

  EXPECT_LT(loss_after, loss_before);
}

TEST(ModelTest, RunTestEdgeCases) {
  s21::MatrixModel model(2, 64);

  s21::Sample s;

  s.label = 0;

  s.data = std::vector<double>(s21::kInputSize, 0.1);

  std::vector<s21::Sample> data = {s};

  EXPECT_NO_THROW(model.RunTest(data, 0.0));

  EXPECT_NO_THROW(model.RunTest(data, 1.0));

  EXPECT_NO_THROW(model.RunTest(data, 10.0));
}

TEST(ModelStructureTest, DynamicLayerChange) {
  s21::MatrixModel model_m(2, 128);
  EXPECT_NO_THROW(model_m.SetLayers(5));

  s21::GraphModel model_g(3, 128);
  EXPECT_NO_THROW(model_g.SetLayers(4));
}

TEST(GraphModelTest, SaveLoadIntegrity) {
  std::vector<double> input(s21::kInputSize, 0.1);
  std::string filename = "test_graph_weights.bin";

  s21::GraphModel model_a(2, 64);
  auto out_a = model_a.Predict(input);
  model_a.SaveWeights(filename);

  s21::GraphModel model_b(2, 64);
  model_b.LoadWeights(filename);
  auto out_b = model_b.Predict(input);

  for (size_t i = 0; i < s21::kOutputSize; ++i) {
    EXPECT_NEAR(out_a[i], out_b[i], 1e-6);
  }
  std::remove(filename.c_str());
}

TEST(ModelExceptionTest, LoadNonExistentFile) {
  s21::MatrixModel m(2, 64);
  s21::GraphModel g(2, 64);

  EXPECT_ANY_THROW(m.LoadWeights("/non/existent/path/file.bin"));
  EXPECT_ANY_THROW(g.LoadWeights("/non/existent/path/file.bin"));
}

TEST(MetricsTest, ZeroSamples) {
  std::vector<int> empty_p, empty_t;
  s21::Metrics m = s21::MetricsCollector::Calculate(empty_p, empty_t);
  EXPECT_DOUBLE_EQ(m.accuracy, 0.0);
}

TEST(LossTest, EmptyDataset) {
  s21::MatrixModel model(2, 64);

  std::vector<s21::Sample> empty;

  EXPECT_DOUBLE_EQ(model.CalculateCrossEntropy(empty), 0.0);
}

TEST(LossTest, NumericalStability) {
  s21::MatrixModel model(2, 64);

  s21::Sample s;

  s.label = 0;

  s.data = std::vector<double>(s21::kInputSize, -1000.0);

  std::vector<s21::Sample> data = {s};

  EXPECT_NO_THROW(model.CalculateCrossEntropy(data));
}

TEST(ModelTest, RunTestClamp) {
  s21::MatrixModel model(2, 64);

  s21::Sample sample;
  sample.label = 0;
  sample.data = std::vector<double>(s21::kInputSize, 0.5);

  std::vector<s21::Sample> data = {sample};

  EXPECT_NO_THROW(model.RunTest(data, -10.0));

  EXPECT_NO_THROW(model.RunTest(data, 100.0));
}

TEST(MatrixModelTest, InvalidInputSize) {
  s21::MatrixModel model(2, 64);

  std::vector<double> bad_input(10);

  EXPECT_ANY_THROW(model.Predict(bad_input));
}

TEST(MetricsTest, CompletelyWrongPredictions) {
  std::vector<int> pred = {0, 0, 0, 0};

  std::vector<int> target = {1, 1, 1, 1};

  auto m = s21::MetricsCollector::Calculate(pred, target);

  EXPECT_DOUBLE_EQ(m.accuracy, 0.0);
}

TEST(ModelStructureTest, DifferentLayerCounts) {
  EXPECT_NO_THROW(s21::MatrixModel(2, 32));

  EXPECT_NO_THROW(s21::MatrixModel(5, 128));

  EXPECT_NO_THROW(s21::GraphModel(2, 32));

  EXPECT_NO_THROW(s21::GraphModel(5, 128));
}
