#pragma once
#include <fstream>
#include <string>
// #include <utility>
#include <vector>

#include "model_types.h"

namespace s21 {

struct KFoldSplit {
  std::vector<Sample> train;
  std::vector<Sample> test;
};

class DatasetReader {
 public:
  static std::vector<Sample> LoadDataset(const std::string& path,
                                         double limit_percent = 1.0) {
    std::vector<Sample> result;
    std::ifstream file(path);
    if (!file.is_open()) return result;

    result.reserve(limit_percent < 1.0 ? 1000 : 130000);

    std::string line;
    while (std::getline(file, line)) {
      if (line.empty() || line.length() < 10) continue;

      Sample s;
      s.data.assign(kInputSize, 0.0);
      const char* ptr = line.c_str();
      int current_val = 0, col_index = -1;

      while (*ptr) {
        if (*ptr == ',') {
          if (col_index == -1)
            s.label = current_val - 1;
          else if (col_index < (int)kInputSize)
            s.data[col_index] = current_val / 255.0;
          current_val = 0;
          col_index++;
        } else if (*ptr >= '0' && *ptr <= '9') {
          current_val = current_val * 10 + (*ptr - '0');
        }
        ptr++;
      }
      if (col_index >= 0 && col_index < (int)kInputSize)
        s.data[col_index] = current_val / 255.0;

      std::vector<double> rotated(kInputSize);
      for (int i = 0; i < 28; ++i)
        for (int j = 0; j < 28; ++j) rotated[i * 28 + j] = s.data[j * 28 + i];
      s.data = std::move(rotated);

      result.push_back(std::move(s));
      if (limit_percent < 1.0 &&
          result.size() >= (size_t)(120000 * limit_percent))
        break;
    }
    return result;
  }

  static std::vector<KFoldSplit> BuildKFold(const std::vector<Sample>& data,
                                            size_t k) {
    std::vector<KFoldSplit> result;

    if (k == 0 || data.empty()) return result;

    size_t chunk = data.size() / k;
    for (size_t current_group = 0; current_group < k; ++current_group) {
      auto test_start = data.begin() + current_group * chunk;
      auto test_end =
          (current_group == k - 1) ? data.end() : test_start + chunk;
      KFoldSplit split;
      split.test.assign(test_start, test_end);
      split.train.reserve(data.size() - split.test.size());
      split.train.insert(split.train.end(), data.begin(), test_start);
      split.train.insert(split.train.end(), test_end, data.end());
      result.push_back(std::move(split));
    }
    return result;
  }
};

}  // namespace s21