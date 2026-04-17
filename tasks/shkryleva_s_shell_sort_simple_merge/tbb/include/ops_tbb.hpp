#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shkryleva_s_shell_sort_simple_merge {

class ShkrylevaSShellMergeTBB : public ppc::core::Task {
 public:
  explicit ShkrylevaSShellMergeTBB(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void ShellSort(int left, int right, std::vector<int> &arr);
  static void Merge(int left, int mid, int right, std::vector<int> &arr, std::vector<int> &buffer);

  std::vector<int> input_;
  std::vector<int> output_;
};

}  // namespace shkryleva_s_shell_sort_simple_merge
