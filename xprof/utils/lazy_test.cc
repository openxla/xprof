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

#include "xprof/utils/lazy.h"

#include <atomic>
#include <memory>
#include <stdexcept>  // IWYU pragma: keep
#include <thread>     // NOLINT
#include <type_traits>
#include <utility>
#include <vector>

#include "<gtest/gtest.h>"
#include "absl/base/config.h"  // IWYU pragma: keep
#include "absl/synchronization/notification.h"

namespace xprof {
namespace {

// Verify type traits.
static_assert(!std::is_copy_constructible_v<ThreadSafeLazy<int>>);
static_assert(!std::is_copy_assignable_v<ThreadSafeLazy<int>>);
static_assert(std::is_move_constructible_v<ThreadSafeLazy<int>>);
static_assert(std::is_move_assignable_v<ThreadSafeLazy<int>>);
static_assert(std::is_nothrow_move_constructible_v<ThreadSafeLazy<int>>);
static_assert(std::is_nothrow_move_assignable_v<ThreadSafeLazy<int>>);

struct ThrowingMove {
  ThrowingMove() = default;
  ThrowingMove(ThrowingMove&&) noexcept(false) {}
  ThrowingMove& operator=(ThrowingMove&&) noexcept(false) { return *this; }
};
static_assert(
    !std::is_nothrow_move_constructible_v<ThreadSafeLazy<ThrowingMove>>);
static_assert(!std::is_nothrow_move_assignable_v<ThreadSafeLazy<ThrowingMove>>);

class InstanceTracker {
 public:
  explicit InstanceTracker(int* active_count) : active_count_(active_count) {
    (*active_count_)++;
  }
  ~InstanceTracker() { (*active_count_)--; }

  // Non-copyable, non-movable to ensure ThreadSafeLazy works with strict types.
  InstanceTracker(const InstanceTracker&) = delete;
  InstanceTracker& operator=(const InstanceTracker&) = delete;
  InstanceTracker(InstanceTracker&&) = delete;
  InstanceTracker& operator=(InstanceTracker&&) = delete;

  int GetValue() const { return 42; }

 private:
  int* active_count_;
};

TEST(ThreadSafeLazyTest, BasicInitialization) {
  int call_count = 0;
  const ThreadSafeLazy<int> lazy([&call_count]() {
    ++call_count;
    return 42;
  });

  EXPECT_EQ(call_count, 0);
  EXPECT_EQ(lazy.Get(), 42);
  EXPECT_EQ(call_count, 1);
  EXPECT_EQ(lazy.Get(), 42);
  EXPECT_EQ(call_count, 1);
}

TEST(ThreadSafeLazyTest, LifecycleAndComplexTypes) {
  int active_instances = 0;
  {
    const ThreadSafeLazy<std::unique_ptr<InstanceTracker>> lazy(
        [&active_instances]() {
          return std::make_unique<InstanceTracker>(&active_instances);
        });

    EXPECT_EQ(active_instances, 0);
    EXPECT_EQ(lazy.Get()->GetValue(), 42);
    EXPECT_EQ(active_instances, 1);
    EXPECT_EQ(lazy.Get()->GetValue(), 42);
    EXPECT_EQ(active_instances, 1);
  }
  EXPECT_EQ(active_instances, 0);
}

TEST(ThreadSafeLazyTest, UninitializedDestruction) {
  int active_instances = 0;
  {
    const ThreadSafeLazy<std::unique_ptr<InstanceTracker>> lazy(
        [&active_instances]() {
          return std::make_unique<InstanceTracker>(&active_instances);
        });
    EXPECT_EQ(active_instances, 0);
  }
  EXPECT_EQ(active_instances, 0);
}

TEST(ThreadSafeLazyTest, ReleasesCaptureMemoryUponEvaluation) {
  int capture_instances = 0;
  auto tracker = std::make_shared<InstanceTracker>(&capture_instances);
  EXPECT_EQ(capture_instances, 1);

  const ThreadSafeLazy<int> lazy(
      [tracker = std::move(tracker)]() { return tracker->GetValue(); });
  EXPECT_EQ(capture_instances, 1);

  EXPECT_EQ(lazy.Get(), 42);
  // After Get(), the initializer closure should have been reset.
  EXPECT_EQ(capture_instances, 0);
}

TEST(ThreadSafeLazyTest, ThreadSafety) {
  std::atomic<int> call_count{0};
  const ThreadSafeLazy<int> lazy([&call_count]() {
    ++call_count;
    return 100;
  });

  absl::Notification start_notification;
  {
    std::vector<std::jthread> threads;
    threads.reserve(10);
    for (int i = 0; i < 10; ++i) {
      threads.emplace_back([&lazy, &start_notification]() {
        start_notification.WaitForNotification();
        EXPECT_EQ(lazy.Get(), 100);
      });
    }

    // Unleash all threads simultaneously to maximize contention.
    start_notification.Notify();
  }  // All jthreads join upon scope exit.

  EXPECT_EQ(call_count.load(), 1);
}

TEST(ThreadSafeLazyTest, MoveConstructibleBeforeInit) {
  int call_count = 0;
  ThreadSafeLazy<int> lazy1([&call_count]() {
    ++call_count;
    return 42;
  });

  const ThreadSafeLazy<int> lazy2(std::move(lazy1));

  EXPECT_EQ(call_count, 0);
  EXPECT_EQ(lazy2.Get(), 42);
  EXPECT_EQ(call_count, 1);
  EXPECT_EQ(lazy2.Get(), 42);
  EXPECT_EQ(call_count, 1);
}

TEST(ThreadSafeLazyTest, MoveConstructibleAfterInit) {
  int call_count = 0;
  ThreadSafeLazy<int> lazy1([&call_count]() {
    ++call_count;
    return 42;
  });

  EXPECT_EQ(lazy1.Get(), 42);
  EXPECT_EQ(call_count, 1);

  const ThreadSafeLazy<int> lazy2(std::move(lazy1));

  EXPECT_EQ(lazy2.Get(), 42);
  EXPECT_EQ(call_count, 1);  // Should NOT re-evaluate
}

TEST(ThreadSafeLazyTest, MoveAssignableBeforeInit) {
  int call_count1 = 0;
  int call_count2 = 0;
  ThreadSafeLazy<int> lazy1([&call_count1]() {
    ++call_count1;
    return 11;
  });

  ThreadSafeLazy<int> lazy2([&call_count2]() {
    ++call_count2;
    return 22;
  });

  lazy2 = std::move(lazy1);

  EXPECT_EQ(call_count1, 0);
  EXPECT_EQ(call_count2, 0);
  EXPECT_EQ(lazy2.Get(), 11);
  EXPECT_EQ(call_count1, 1);
  EXPECT_EQ(call_count2, 0);
}

TEST(ThreadSafeLazyTest, MoveAssignableAfterInit) {
  int call_count1 = 0;
  int call_count2 = 0;
  ThreadSafeLazy<int> lazy1([&call_count1]() {
    ++call_count1;
    return 11;
  });
  EXPECT_EQ(lazy1.Get(), 11);
  EXPECT_EQ(call_count1, 1);

  ThreadSafeLazy<int> lazy2([&call_count2]() {
    ++call_count2;
    return 22;
  });

  lazy2 = std::move(lazy1);

  EXPECT_EQ(lazy2.Get(), 11);
  EXPECT_EQ(call_count1, 1);  // Should NOT re-evaluate
  EXPECT_EQ(call_count2, 0);
}

TEST(ThreadSafeLazyTest, MoveAssignableDestInitSourceNotInit) {
  int call_count1 = 0;
  int call_count2 = 0;
  ThreadSafeLazy<int> lazy1([&call_count1]() {
    ++call_count1;
    return 11;
  });

  ThreadSafeLazy<int> lazy2([&call_count2]() {
    ++call_count2;
    return 22;
  });
  EXPECT_EQ(lazy2.Get(), 22);
  EXPECT_EQ(call_count2, 1);

  lazy2 = std::move(lazy1);

  EXPECT_EQ(lazy2.Get(), 11);
  EXPECT_EQ(call_count1, 1);
  EXPECT_EQ(call_count2, 1);
}

TEST(ThreadSafeLazyTest, MoveAssignableDestInitSourceInit) {
  int call_count1 = 0;
  int call_count2 = 0;
  ThreadSafeLazy<int> lazy1([&call_count1]() {
    ++call_count1;
    return 11;
  });
  EXPECT_EQ(lazy1.Get(), 11);
  EXPECT_EQ(call_count1, 1);

  ThreadSafeLazy<int> lazy2([&call_count2]() {
    ++call_count2;
    return 22;
  });
  EXPECT_EQ(lazy2.Get(), 22);
  EXPECT_EQ(call_count2, 1);

  lazy2 = std::move(lazy1);

  EXPECT_EQ(lazy2.Get(), 11);
  EXPECT_EQ(call_count1, 1);
  EXPECT_EQ(call_count2, 1);
}

#ifdef ABSL_HAVE_EXCEPTIONS
TEST(ThreadSafeLazyTest, ExceptionHandlingAndRetry) {
  int call_count = 0;
  bool should_throw = true;
  const ThreadSafeLazy<int> lazy([&call_count, &should_throw]() {
    ++call_count;
    if (should_throw) {
      throw std::runtime_error("Initialization failed");
    }
    return 42;
  });

  EXPECT_EQ(call_count, 0);
  EXPECT_THROW(lazy.Get(), std::runtime_error);
  EXPECT_EQ(call_count, 1);

  // The instance should remain uninitialized and subsequent Get() should retry.
  should_throw = false;
  EXPECT_EQ(lazy.Get(), 42);
  EXPECT_EQ(call_count, 2);

  // Once initialized, subsequent Get() calls return the cached value.
  EXPECT_EQ(lazy.Get(), 42);
  EXPECT_EQ(call_count, 2);
}
#endif  // ABSL_HAVE_EXCEPTIONS

}  // namespace
}  // namespace xprof
