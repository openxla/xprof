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

#ifndef XPROF_UTILS_LAZY_H_
#define XPROF_UTILS_LAZY_H_

#include <atomic>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"

namespace xprof {

// Evaluates an initializer exactly once and caches the result.
//
// This class is thread-safe for concurrent access via `Get()`. Multiple threads
// can concurrently call `Get()`, and the class ensures that the provided
// Initializer is invoked exactly once. Concurrent callers will block until the
// initialization completes, after which all callers will receive a reference
// to the same cached value.
//
// Note: Modifying, moving, or destroying an instance while other threads are
// concurrently calling `Get()` is not thread-safe and requires external
// synchronization (standard C++ object lifecycle semantics).
//
// Note: Initializer callables must not call `Get()` on the same
// `ThreadSafeLazy` instance (directly or indirectly); doing so will result in a
// deadlock because the underlying mutex is not recursive.
//
// Note: The `ThreadSafeLazy` container itself requires 0 heap allocations on
// construction. However, `absl::AnyInvocable` may allocate memory on the heap
// if the captured state of the initializer callable exceeds its inline storage
// capacity (Small Object Optimization limit). Keep lambda captures small (e.g.
// capturing by reference or pointer when safe) to ensure zero heap allocations.
//
// Example:
// ```cpp
// const ThreadSafeLazy<std::string> lazy_str([] {
//   return "expensive initialization";
// });
//
// // In any thread:
// const std::string& value = lazy_str.Get();
// ```
template <typename T>
class ThreadSafeLazy {
 public:
  using Initializer = absl::AnyInvocable<T()>;

  explicit ThreadSafeLazy(Initializer init) : init_(std::move(init)) {}

  // Move constructor: transfers state or cached value.
  ThreadSafeLazy(ThreadSafeLazy&& other) noexcept(
      std::is_nothrow_move_constructible_v<T>)
      : value_(std::move(other.value_)),
        init_(std::move(other.init_)),
        initialized_(other.initialized_.load(std::memory_order_relaxed)) {}

  // Move assignment operator.
  ThreadSafeLazy& operator=(ThreadSafeLazy&& other) noexcept(
      std::is_nothrow_move_constructible_v<T> &&
      std::is_nothrow_move_assignable_v<T>) {
    if (this != &other) {
      value_ = std::move(other.value_);
      init_ = std::move(other.init_);
      initialized_.store(other.initialized_.load(std::memory_order_relaxed),
                         std::memory_order_relaxed);
    }
    return *this;
  }

  // Non-copyable.
  ThreadSafeLazy(const ThreadSafeLazy&) = delete;
  ThreadSafeLazy& operator=(const ThreadSafeLazy&) = delete;

  const T& Get() const {
    if (ABSL_PREDICT_FALSE(!initialized_.load(std::memory_order_acquire))) {
      InitSlow();
    }
    return *value_;
  }

 private:
  ABSL_ATTRIBUTE_NOINLINE void InitSlow() const {
    absl::MutexLock lock(mu_);
    if (!initialized_.load(std::memory_order_relaxed)) {
      value_ = init_();
      init_ = nullptr;  // Release any captured resources in the initializer.
      initialized_.store(true, std::memory_order_release);
    }
  }

  mutable absl::Mutex mu_;
  mutable std::optional<T> value_;
  mutable Initializer init_;
  mutable std::atomic<bool> initialized_{false};
};

}  // namespace xprof

#endif  // XPROF_UTILS_LAZY_H_
