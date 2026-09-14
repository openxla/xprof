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

#ifndef THIRD_PARTY_XPROF_CONVERT_EXECUTOR_FACTORY_H_
#define THIRD_PARTY_XPROF_CONVERT_EXECUTOR_FACTORY_H_

#include <memory>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "xprof/convert/executor.h"

namespace tensorflow::profiler {

// Non-owning callable reference that returns a newly created Executor instance.
// Functions that receive this reference as an argument must not store it beyond
// the immediate function call.
using ExecutorFactoryRef = absl::FunctionRef<std::unique_ptr<Executor>()>;

// Returns an `Executor` instance that runs tasks synchronously on the calling
// thread without spawning worker threads.
std::unique_ptr<Executor> InlineExecutorFactory();

// Creates an `XprofThreadPoolExecutor` instance with the specified `name`.
// Consult `XprofThreadPoolExecutor` for details on the default thread count.
std::unique_ptr<Executor> CreateXprofThreadPoolExecutor(absl::string_view name);

// Creates an `XprofThreadPoolExecutor` instance with the specified `name` and
// `num_threads`. Consult `XprofThreadPoolExecutor` for details on the
// fallback behavior if `num_threads` is 0 or negative.
std::unique_ptr<Executor> CreateXprofThreadPoolExecutor(absl::string_view name,
                                                       int num_threads);

// Equivalent to `CreateXprofThreadPoolExecutor("xprof_executor")`.
std::unique_ptr<Executor> DefaultExecutorFactory();

}  // namespace tensorflow::profiler

#endif  // THIRD_PARTY_XPROF_CONVERT_EXECUTOR_FACTORY_H_
