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

#include <memory>

#include "<gtest/gtest.h>"
#include "xprof/convert/executor.h"

namespace tensorflow::profiler {
namespace {

TEST(ExecutorFactoryTest, DefaultExecutorFactoryCreatesValidExecutor) {
  std::unique_ptr<Executor> executor = DefaultExecutorFactory();
  ASSERT_NE(executor, nullptr);

  bool executed = false;
  executor->Execute([&] { executed = true; });
  executor->JoinAll();
  EXPECT_TRUE(executed);
}

TEST(ExecutorFactoryTest, CreateXprofThreadPoolExecutorWithDefaultThreads) {
  std::unique_ptr<Executor> executor =
      CreateXprofThreadPoolExecutor("test_pool_default_threads");
  ASSERT_NE(executor, nullptr);

  bool executed = false;
  executor->Execute([&] { executed = true; });
  executor->JoinAll();
  EXPECT_TRUE(executed);
}

TEST(ExecutorFactoryTest, CreateXprofThreadPoolExecutorWithExplicitThreads) {
  std::unique_ptr<Executor> executor =
      CreateXprofThreadPoolExecutor("test_pool_explicit_threads", 4);
  ASSERT_NE(executor, nullptr);

  bool executed = false;
  executor->Execute([&] { executed = true; });
  executor->JoinAll();
  EXPECT_TRUE(executed);
}

TEST(ExecutorFactoryTest, CreateXprofThreadPoolExecutorWithZeroThreads) {
  std::unique_ptr<Executor> executor =
      CreateXprofThreadPoolExecutor("test_pool_zero_threads", 0);
  ASSERT_NE(executor, nullptr);

  bool executed = false;
  executor->Execute([&] { executed = true; });
  executor->JoinAll();
  EXPECT_TRUE(executed);
}

TEST(ExecutorFactoryTest, InlineExecutorFactoryExecutesSynchronously) {
  std::unique_ptr<Executor> executor = InlineExecutorFactory();
  ASSERT_NE(executor, nullptr);

  bool executed = false;
  executor->Execute([&] { executed = true; });
  EXPECT_TRUE(executed);
  executor->JoinAll();
}

TEST(ExecutorFactoryTest, ExecutorFactoryRefWorks) {
  auto run_with_factory = [](ExecutorFactoryRef factory) {
    std::unique_ptr<Executor> executor = factory();
    bool executed = false;
    executor->Execute([&] { executed = true; });
    executor->JoinAll();
    return executed;
  };

  EXPECT_TRUE(run_with_factory(DefaultExecutorFactory));
  EXPECT_TRUE(run_with_factory(InlineExecutorFactory));
}

}  // namespace
}  // namespace tensorflow::profiler
