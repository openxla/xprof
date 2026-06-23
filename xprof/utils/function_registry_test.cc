/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/utils/function_registry.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {
namespace {

using ::testing::AllOf;
using ::testing::IsTrue;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

ABSL_CONST_INIT const absl::string_view kKey = "key";

TEST(FunctionRegistryTest, RegisterSucceeds) {
  FunctionRegistry<absl::string_view, int()> registry;
  EXPECT_TRUE(registry.Register("foo", [] { return 1; }));
  RegisterOrDie(&registry, "bar", [] { return 2; });
}

struct NonStringifiable {
  int value;

  template <typename H>
  friend H AbslHashValue(H h, const NonStringifiable& n) {
    return H::combine(std::move(h), n.value);
  }

  friend bool operator==(const NonStringifiable& a, const NonStringifiable& b) {
    return a.value == b.value;
  }
};

TEST(FunctionRegistryTest, RegisterWorksWithNonStringifiableKey) {
  FunctionRegistry<NonStringifiable, int()> registry;
  EXPECT_TRUE(registry.Register({.value = 1}, [] { return 1; }));
}

TEST(FunctionRegistryTest, HeterogeneousRegisterSucceeds) {
  FunctionRegistry<std::string, int()> registry;
  EXPECT_TRUE(registry.Register(
      kKey, [] { return 1; }));
  absl::string_view bar = "bar";
  RegisterOrDie(registry, bar, [] { return 2; });
}

TEST(FunctionRegistryTest, RegisterReferenceSucceeds) {
  FunctionRegistry<absl::string_view, int()> registry;
  EXPECT_TRUE(registry.Register("foo", [] { return 1; }));
  RegisterOrDie(registry, "bar", [] { return 2; });
}

TEST(FunctionRegistryTest, RegisterFails) {
  FunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register(kKey, [] { return 1; }));

  // Try to re-register the same key.
  EXPECT_FALSE(registry.Register(kKey, [] { return 2; }));
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(RegisterOrDie(
                   &registry, kKey, [] { return 2; }),
               "Registration failed.*");
#endif
}

TEST(FunctionRegistryTest, Unregister) {
  FunctionRegistry<absl::string_view, int()> registry;
  registry.Register(kKey, [] { return 1; });
  registry.Unregister(kKey);

  // Now we can re-register the same key.
  EXPECT_TRUE(registry.Register(kKey, [] { return 2; }));
}

TEST(FunctionRegistryTest, GetSucceeds) {
  // Use a move-only function argument to test forwarding.
  FunctionRegistry<absl::string_view, int(std::unique_ptr<int>)>  // NOLINT
      registry;
  ASSERT_TRUE(
      registry.Register(kKey, [](std::unique_ptr<int> x) { return *x; }));

  auto function = registry.Get(kKey);
  ASSERT_TRUE(function);
  EXPECT_EQ(1, function(std::make_unique<int>(1)));
}

TEST(FunctionRegistryTest, GetFails) {
  FunctionRegistry<absl::string_view, int()> registry;
  auto function = registry.Get(kKey);
  EXPECT_FALSE(function);
}

MATCHER_P(WhenInvokedEquals, expected, "") {
  auto actual = arg();
  *result_listener << "Expected object named '" << expected << "', "
                   << "got '" << actual << "'";
  return expected == actual;
}

TEST(FunctionRegistryTest, GetAll) {
  FunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register("foo", [] { return 1; }));
  ASSERT_TRUE(registry.Register("bar", [] { return 2; }));

  auto functions = registry.GetAll();
  EXPECT_THAT(
      functions,
      UnorderedElementsAre(Pair("foo", AllOf(IsTrue(), WhenInvokedEquals(1))),
                           Pair("bar", AllOf(IsTrue(), WhenInvokedEquals(2)))));
}

TEST(FunctionRegistryTest, FunctionLifetime) {
  FunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register(kKey, [] { return 1; }));

  auto function = registry.Get(kKey);
  registry.Unregister(kKey);

  // Even though the key is unregistered, we can still use the std::function.
  ASSERT_TRUE(function);
  EXPECT_EQ(1, function());
}

TEST(FunctionRegistryTest, FunctionCopy) {
  FunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register(kKey, [] { return 1; }));

  std::function<int()> fn = registry.Get(kKey);
  ASSERT_TRUE(fn);

  auto copy = fn;
  ASSERT_TRUE(copy);

  EXPECT_EQ(1, fn());
  EXPECT_EQ(1, copy());
}

TEST(FunctionRegistryTest, FunctionMove) {
  FunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register(kKey, [] { return 1; }));

  std::function<int()> fn = registry.Get(kKey);
  ASSERT_TRUE(fn);

  auto move = std::move(fn);
  ASSERT_TRUE(move);
  EXPECT_EQ(1, move());
}

TEST(FunctionRegistryTest, StatefulFunctor) {
  struct Counter {
    int n = 1;
    int operator()() { return n++; }
  };

  FunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register(kKey, Counter{}));

  auto fn = registry.Get(kKey);
  EXPECT_EQ(1, fn());
  EXPECT_EQ(2, fn());

  // Get returns a reference, technically.
  fn = registry.Get(kKey);
  EXPECT_EQ(3, fn());
}

TEST(FunctionRegistryTest, UnsetFunction) {
  FunctionRegistry<absl::string_view, int()> registry;
  std::function<int()> original_fn;
  EXPECT_FALSE(!!original_fn);
  ASSERT_TRUE(registry.Register(kKey, original_fn));

  auto fn = registry.Get(kKey);
  EXPECT_FALSE(!!fn);
}

TEST(FunctionRegistryTest, ScopedRegistration) {
  FunctionRegistry<absl::string_view, int()> registry;
  EXPECT_FALSE(registry.Get("foo"));
  {
    ScopedRegistration registration(registry, "foo", [] { return 1; });
    EXPECT_TRUE(registry.Get("foo"));
  }
  // "foo" was unregistered at the end of the scope
  EXPECT_FALSE(registry.Get("foo"));
}

TEST(FunctionRegistryTest, ScopedRegistrationMoveConstructor) {
  FunctionRegistry<absl::string_view, int()> registry;
  ScopedRegistration bar_registration(registry, "bar", [] { return 2; });
  EXPECT_TRUE(registry.Get("bar"));
  {
    ScopedRegistration bar_registration2 = std::move(bar_registration);
    EXPECT_TRUE(registry.Get("bar"));
  }
  // "bar" shouldn't be registered anymore
  EXPECT_FALSE(registry.Get("bar"));
}

TEST(FunctionRegistryTest, ScopedRegistrationMoveAssignment) {
  FunctionRegistry<absl::string_view, int()> registry;
  ScopedRegistration bar_registration(registry, "bar", [] { return 2; });
  EXPECT_FALSE(registry.Get("foo"));
  EXPECT_TRUE(registry.Get("bar"));
  {
    ScopedRegistration foo_registration(registry, "foo", [] { return 1; });
    bar_registration = std::move(foo_registration);
    // "foo" should still be registered, "bar" should no longer be
    EXPECT_TRUE(registry.Get("foo"));
    EXPECT_FALSE(registry.Get("bar"));
  }
  // "foo" should still be registered after the scope, as it was moved.
  EXPECT_TRUE(registry.Get("foo"));
}

#if GTEST_HAS_DEATH_TEST
TEST(FunctionRegistryTest, ScopedRegistrationDuplicate) {
  FunctionRegistry<absl::string_view, int()> registry;
  registry.Register("foo", [] { return 1; });
  EXPECT_TRUE(registry.Get("foo"));
  EXPECT_DEATH(
      ScopedRegistration registration(registry, "foo", [] { return 1; }),
      "Registration failed.*");
}
#endif

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
