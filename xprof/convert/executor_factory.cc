/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xprof/convert/executor_factory.h"

#include <functional>
#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "xprof/convert/executor.h"
#include "xprof/convert/xprof_thread_pool_executor.h"

namespace tensorflow::profiler {
namespace {

// Executes tasks synchronously on the calling thread.
class InlineExecutor : public Executor {
 public:
  void Execute(std::function<void()> fn) override { fn(); }
  void JoinAll() override {}
};

}  // namespace

std::unique_ptr<Executor> InlineExecutorFactory() {
  return std::make_unique<InlineExecutor>();
}

std::unique_ptr<Executor> CreateXprofThreadPoolExecutor(
    absl::string_view name) {
  return std::make_unique<XprofThreadPoolExecutor>(std::string(name));
}

std::unique_ptr<Executor> CreateXprofThreadPoolExecutor(
    absl::string_view name, int num_threads) {
  return std::make_unique<XprofThreadPoolExecutor>(std::string(name),
                                                   num_threads);
}

std::unique_ptr<Executor> DefaultExecutorFactory() {
  return CreateXprofThreadPoolExecutor("xprof_executor");
}

}  // namespace tensorflow::profiler
