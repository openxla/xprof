#include "xprof/utils/function_registry.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "testing/base/public/gmock.h"
#include "<gtest/gtest.h>"
#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "absl/types/source_location.h"

namespace util_registration {
namespace {

using ::testing::AllOf;
using ::testing::IsTrue;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

ABSL_CONST_INIT const absl::string_view kKey = "key";

TEST(XprofFunctionRegistryTest, RegisterSucceeds) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  EXPECT_TRUE(registry.Register("foo", [] { return 1; }, ABSL_LOC));
  XprofRegisterOrDie(&registry, "bar", [] { return 2; }, ABSL_LOC);
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

TEST(XprofFunctionRegistryTest, RegisterWorksWithNonStringifiableKey) {
  XprofFunctionRegistry<NonStringifiable, int()> registry;
  EXPECT_TRUE(registry.Register({.value = 1}, [] { return 1; }, ABSL_LOC));
}

TEST(XprofFunctionRegistryTest, HeterogeneousRegisterSucceeds) {
  XprofFunctionRegistry<std::string, int()> registry;
  EXPECT_TRUE(registry.Register(
      kKey, [] { return 1; }, ABSL_LOC));
  absl::string_view bar = "bar";
  XprofRegisterOrDie(registry, bar, [] { return 2; }, ABSL_LOC);
}

TEST(XprofFunctionRegistryTest, RegisterReferenceSucceeds) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  EXPECT_TRUE(registry.Register("foo", [] { return 1; }, ABSL_LOC));
  XprofRegisterOrDie(registry, "bar", [] { return 2; }, ABSL_LOC);
}

TEST(XprofFunctionRegistryTest, RegisterFails) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register(kKey, [] { return 1; }, ABSL_LOC));

  // Try to re-register the same key.
  EXPECT_FALSE(registry.Register(kKey, [] { return 2; }, ABSL_LOC));
#if GTEST_HAS_DEATH_TEST
  EXPECT_DEATH(XprofRegisterOrDie(
                   &registry, kKey, [] { return 2; }, ABSL_LOC),
               "Registration failed.*");
#endif
}

TEST(XprofFunctionRegistryTest, Unregister) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  registry.Register(kKey, [] { return 1; }, ABSL_LOC);
  registry.Unregister(kKey);

  // Now we can re-register the same key.
  EXPECT_TRUE(registry.Register(kKey, [] { return 2; }, ABSL_LOC));
}

TEST(XprofFunctionRegistryTest, GetSucceeds) {
  // Use a move-only function argument to test forwarding.
  XprofFunctionRegistry<absl::string_view, int(std::unique_ptr<int>)>  // NOLINT
      registry;
  ASSERT_TRUE(registry.Register(kKey, [](std::unique_ptr<int> x) { return *x; },
                                ABSL_LOC));

  auto function = registry.Get(kKey);
  ASSERT_TRUE(function);
  EXPECT_EQ(1, function(std::make_unique<int>(1)));
}

TEST(XprofFunctionRegistryTest, GetFails) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  auto function = registry.Get(kKey);
  EXPECT_FALSE(function);
}

MATCHER_P(WhenInvokedEquals, expected, "") {
  auto actual = arg();
  *result_listener << "Expected object named '" << expected << "', "
                   << "got '" << actual << "'";
  return expected == actual;
}

TEST(XprofFunctionRegistryTest, GetAll) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register("foo", [] { return 1; }, ABSL_LOC));
  ASSERT_TRUE(registry.Register("bar", [] { return 2; }, ABSL_LOC));

  auto functions = registry.GetAll();
  EXPECT_THAT(
      functions,
      UnorderedElementsAre(Pair("foo", AllOf(IsTrue(), WhenInvokedEquals(1))),
                           Pair("bar", AllOf(IsTrue(), WhenInvokedEquals(2)))));
}

TEST(XprofFunctionRegistryTest, FunctionLifetime) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register(kKey, [] { return 1; }, ABSL_LOC));

  auto function = registry.Get(kKey);
  registry.Unregister(kKey);

  // Even though the key is unregistered, we can still use the std::function.
  ASSERT_TRUE(function);
  EXPECT_EQ(1, function());
}

TEST(XprofFunctionRegistryTest, FunctionCopy) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register(kKey, [] { return 1; }, ABSL_LOC));

  std::function<int()> fn = registry.Get(kKey);
  ASSERT_TRUE(fn);

  auto copy = fn;
  ASSERT_TRUE(copy);

  EXPECT_EQ(1, fn());
  EXPECT_EQ(1, copy());
}

TEST(XprofFunctionRegistryTest, FunctionMove) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register(kKey, [] { return 1; }, ABSL_LOC));

  std::function<int()> fn = registry.Get(kKey);
  ASSERT_TRUE(fn);

  auto move = std::move(fn);
  ASSERT_TRUE(move);
  EXPECT_EQ(1, move());
}

TEST(XprofFunctionRegistryTest, StatefulFunctor) {
  struct Counter {
    int n = 1;
    int operator()() { return n++; }
  };

  XprofFunctionRegistry<absl::string_view, int()> registry;
  ASSERT_TRUE(registry.Register(kKey, Counter{}, ABSL_LOC));

  auto fn = registry.Get(kKey);
  EXPECT_EQ(1, fn());
  EXPECT_EQ(2, fn());

  // Get returns a reference, technically.
  fn = registry.Get(kKey);
  EXPECT_EQ(3, fn());
}

TEST(XprofFunctionRegistryTest, UnsetFunction) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  std::function<int()> original_fn;
  EXPECT_FALSE(!!original_fn);
  ASSERT_TRUE(registry.Register(kKey, original_fn, ABSL_LOC));

  auto fn = registry.Get(kKey);
  EXPECT_FALSE(!!fn);
}

TEST(XprofFunctionRegistryTest, XprofScopedRegistration) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  EXPECT_FALSE(registry.Get("foo"));
  {
    XprofScopedRegistration registration(registry, "foo", [] { return 1; });
    EXPECT_TRUE(registry.Get("foo"));
  }
  // "foo" was unregistered at the end of the scope
  EXPECT_FALSE(registry.Get("foo"));
}

TEST(XprofFunctionRegistryTest, XprofScopedRegistrationMoveConstructor) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  XprofScopedRegistration bar_registration(registry, "bar", [] { return 2; });
  EXPECT_TRUE(registry.Get("bar"));
  {
    XprofScopedRegistration bar_registration2 = std::move(bar_registration);
    EXPECT_TRUE(registry.Get("bar"));
  }
  // "bar" shouldn't be registered anymore
  EXPECT_FALSE(registry.Get("bar"));
}

TEST(XprofFunctionRegistryTest, XprofScopedRegistrationMoveAssignment) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  XprofScopedRegistration bar_registration(registry, "bar", [] { return 2; });
  EXPECT_FALSE(registry.Get("foo"));
  EXPECT_TRUE(registry.Get("bar"));
  {
    XprofScopedRegistration foo_registration(registry, "foo", [] { return 1; });
    bar_registration = std::move(foo_registration);
    // "foo" should still be registered, "bar" should no longer be
    EXPECT_TRUE(registry.Get("foo"));
    EXPECT_FALSE(registry.Get("bar"));
  }
  // "foo" should still be registered after the scope, as it was moved.
  EXPECT_TRUE(registry.Get("foo"));
}

#if GTEST_HAS_DEATH_TEST
TEST(XprofFunctionRegistryTest, XprofScopedRegistrationDuplicate) {
  XprofFunctionRegistry<absl::string_view, int()> registry;
  registry.Register("foo", [] { return 1; });
  EXPECT_TRUE(registry.Get("foo"));
  EXPECT_DEATH(
      XprofScopedRegistration registration(registry, "foo", [] { return 1; }),
      "Registration failed.*");
}
#endif

}  // namespace
}  // namespace util_registration
