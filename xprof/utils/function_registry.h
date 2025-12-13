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

// Class that serves as a registry of functions.
//
// === Background =============================================================
//
// A FunctionRegistry maps keys to functions. The key and function types are
// specified in the template arguments.
//
// Here's a simple usage example before we explain the API in more detail:
//
//   // Maps string keys to functions that take two ints and return an int.
//   FunctionRegistry<string, int(int, int)> r;
//
//   // Registers the "add" function
//   r.Register("add", [](int a, int b) { return a + b; });
//
//   // Tries to get "add" from the registry and runs it on success.
//   std::function<int(int, int)> f = r.Get("add");
//   if (f) {
//     int answer = f(4, 2);
//     // answer == 6
//   }
//
// FunctionRegistry allows clients to store objects into some central data
// structure used in a library they don't own. You can think of the registered
// functions as factories for instantiating "plugins". In such a case, the
// registry owner will expose a global registry, and the client will register an
// implementation with it in the dynamic initialization phase of the program. In
// this way, the client does not need to modify the registry owner's code.
//
// A very simple example:
//
// === owner.h ===========
//
// class Foo {
// public:
//   virtual ~Foo() = default;
//   virtual void Do() = 0;
// };
//
// using FooRegistry = FunctionRegistry<string, std::unique_ptr<Foo>()>;
// FooRegistry& GetGlobalFooRegistry();
//
// === owner.cc ==========
//
// FooRegistry& GetGlobalFooRegistry() {
//   static absl::NoDestructor<FooRegistry> r;  // go/totw/110.
//   return *r;
// }
//
// === client.h ==========
//
// class MyFoo : public Foo {
//  public:
//   void Do() override { LOG(INFO) << "Hello world!"; }
// };
//
// === client_registration.cc ==========
//
// #include "client.h"
// #include "owner.h"
//
// const auto kUnused = GetGlobalFooRegistry().Register(
//    "MyFoo", [] { return std::make_unique<MyFoo>(); });
//
// =======================
//
// Now, the registry owner's code can access the registered MyFoo
// implementation at program startup without explicitly needing to include the
// client's code.
//
// The caveat of this approach is that the linker may choose to discard any
// symbols not transitively included in `main`. In this case, kUnused is not
// referenced anywhere in the owner's code, so the registration would be
// discarded. To solve this, make sure to include client_registration.cc in a
// cc_library *by itself* and include "alwayslink = 1" in this cc_library.
//
// cc_library(
//   name = "client_registration",
//   srcs = ["client_registration.cc"],
//   alwayslink = 1,
//   deps = [
//      ":client",
//      "//owner",
//   ],
// )
//
// As Buildozer will warn you, make sure there are no headers in your library
// that includes the registration (go/build-warnings#alwayslink-with-hdrs).
//
// It is recommended that "//client_registration" is included in the deps of the
// owner's cc_binary.
//
// cc_binary(
//   name = "owner_binary",
//   srcs = ["owner_binary.cc"],
//   deps = [
//      ":owner",
//      "//client_registration",
//      "//some_other_registration",
//      ...
//   ],
// )
//
// That way, the owner's global registry is clear of clients' registrations for
// testing. Moreover, the owner may wish to have versions of the binaries with
// different registrations linked in.
//
// === Creating a registry ====================================================
//
// A registry maps keys to functions. The key and function types are
// specified in the template arguments.
//
// FunctionRegistry<string, std::unique_ptr<Foo>(const Env&)> r;
//
// In this particular example, a const Env& (an arbitrary, owner-defined type)
// is an argument to the function. This means all client-defined functions must
// take a const Env&. This may be useful if implementations' constructors rely
// on some context. The arguments to the function can be move-only.
//
// A FunctionRegistry is a regular C++ object: it has a scope and lifetime. If
// you wish to make a global registry, use the recommended way to create any
// global object in go/totw/110.
//
// === Registering functions ==================================================
//
// To register a function, call Register with a key and function. The key must
// be copyable and hashable using go/absl-hash, and the function may be any
// argument convertible to std::function.
//
//   FunctionRegistry<string, std::unique_ptr<Foo>(const Env&)> registry;
//   bool reg = registry.Register(
//       "MyFoo", [](const Env& env) { return std::make_unique<MyFoo>(env); });
//   if (!reg) { HandleError(); }
//
// A call to Register will fail only if the provided key already exists in the
// registry. In this case, an informative error message will be logged about the
// collision.
//
// Registering in global scope is quite similar:
//
//   const bool kUnused = GetGlobalRegistry().Register("MyFoo", &MakeFoo);
//
// If multiple calls to Register in the global scope register the same key,
// it is unspecified which call will succeed and which will fail. The logs will
// indicate which failed due to the collision. To crash the program on
// collision, you can use convenience function RegisterOrDie:
//
//   const bool kUnused =
//      RegisterOrDie(GetGlobalRegistry(), "MyFoo", &MakeFoo);
//
// To unregister an object, call Unregister with a key:
//
//   registry.Unregister("MyFoo");
//
// If an object registration is tied to a particular scope (as in a test), use
// ScopedRegistration to register and unregister the object:
//
// {
//   ScopedRegistration registration(GetGlobalRegistry(), "MyFoo", &MakeFoo);
//   // "MyFoo" is automatically unregistered at the end of the scope
// }
//
// === Getting functions from the registry ====================================
//
// To get a function registered to a particular key, call Get with a key:
//
//   using MyRegistry
//       = FunctionRegistry<string, std::unique_ptr<Foo>(const Env&)>;
//   MyRegistry registry;
//
//   // Prefer using auto in your code, we include the full type for clarity.
//   std::function<std::unique_ptr<Foo>(const Env&)> function
//       = registry.Get("Key");
//   if (!function) { HandleError(); }
//   std::unique_ptr<Foo> foo = function(GetEnv());
//
// To get all entries from the registry, call GetAll:
//
//   for (const auto& e : registry.GetAll()) {
//     const string& key = e.first;
//     const auto& function = e.second;
//     ...
//   }
//
//
// === Testing registries =====================================================
//
// Testing a registry created in non-global scope is trivial, and this is the
// preferred way to test. Ideally, your library dependency injects the registry
// so a scoped registry object can be used in unit tests.
//
// When testing global registries, ensure that no clients' registrations are
// linked in to your test. That way, you have an empty registry in your test.
// As mentioned above, the easiest way to ensure this is to have your clients
// link their registration BUILD rules into your cc_binary rule rather than any
// of your cc_library rules. You can then use Unregister to ensure each test
// starts with a clear registry.
//
// === Thread safety ==========================================================
//
// All operations on the FunctionRegistry are thread safe.

#ifndef THIRD_PARTY_XPROF_UTILS_FUNCTION_REGISTRY_H_
#define THIRD_PARTY_XPROF_UTILS_FUNCTION_REGISTRY_H_

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"

namespace tensorflow {
namespace profiler {
namespace internal {

// Type trait fallback that matches types that cannot be passed to absl::StrCat.
template <typename T, typename = void>
struct IsStrCatable : std::false_type {};

// Type trait that matches types that can be passed to absl::StrCat.
template <typename T>
struct IsStrCatable<
    T, std::enable_if_t<std::is_same_v<
           decltype(absl::StrCat(std::declval<T>())), std::string>>>
    : std::true_type {};

}  // namespace internal

// A registry that maps keys of type K to functions of type Fn.
//
// Example:
//
//   FunctionRegistry<string, int(int, int)> my_registry;
//
// See docs above for more details.
template <typename K, typename Fn>
class FunctionRegistry {
 public:
  using Key = K;
  using Function = std::function<Fn>;

  FunctionRegistry() = default;
  FunctionRegistry(const FunctionRegistry&) = delete;
  FunctionRegistry& operator=(const FunctionRegistry&) = delete;

  // Adds the given key and function to the registry. Returns true if we
  // successfully registered a unique key.
  //
  // See docs above for more details.
  template <typename KeyArg = Key>
  bool Register(
      const KeyArg& key, Function fn) {
    absl::MutexLock lock(&mu_);
    auto insert_result =
        functions_.emplace(key, std::make_shared<MapValue>(std::move(fn)));

    if (!insert_result.second) {
      std::string key_log = "key";
      if constexpr (tensorflow::profiler::internal::IsStrCatable<
                        KeyArg>::value) {
        key_log = absl::StrCat("key '", key, "'");
      }
      LOG(ERROR) << "Registration failed; key already exists in registry, "
                 << key_log << " registered.";
    }

    return insert_result.second;
  }

  // Gets the function associated with key from the registry. If no such key
  // exists, returns a default-constructed (empty) function.
  //
  // See docs above for more details.
  template <typename KeyArg = Key>
  Function Get(const KeyArg& key) const {
    absl::ReaderMutexLock lock(&mu_);
    std::shared_ptr<MapValue> snapshot;
    auto it = functions_.find(key);
    if (it != functions_.end()) {
      snapshot = it->second;
    }
    return ToFunction(std::move(snapshot));
  }

  // Gets all keys and functions from the registry.
  //
  // See docs above for more details.
  std::vector<std::pair<Key, Function>> GetAll() const {
    absl::ReaderMutexLock lock(&mu_);
    std::vector<std::pair<K, Function>> entries;
    for (const auto& kv : functions_) {
      entries.emplace_back(kv.first, ToFunction(kv.second));
    }
    return entries;
  }

  // If the provided key exists in the registry, removes the key and associated
  // function.
  //
  // See docs above for more details.
  template <typename KeyArg = Key>
  void Unregister(const KeyArg& key) {
    absl::MutexLock lock(&mu_);
    functions_.erase(key);
  }

 private:
  struct MapValue {
    MapValue(Function f)
        : func(std::move(f)) {}
    Function func;
  };

  struct FunctionWrapper {
    template <typename... Args>
    typename Function::result_type operator()(Args&&... args) const {
      return snapshot->func(std::forward<Args>(args)...);
    }
    std::shared_ptr<const MapValue> snapshot;
  };

  static Function ToFunction(std::shared_ptr<MapValue> snapshot) {
    if (!snapshot || !snapshot->func) {
      return Function();
    }
    return FunctionWrapper{std::move(snapshot)};
  }

  mutable absl::Mutex mu_;
  absl::flat_hash_map<Key, std::shared_ptr<MapValue>> functions_
      ABSL_GUARDED_BY(mu_);
};

// Adds the given key and function to the registry. CHECK-fails if registration
// fails. The return value is only provided for the convenience of initializing
// a static variable, and is otherwise meaningless.
//
// See docs above for more details.
template <typename Registry, typename Key = typename Registry::Key>
bool RegisterOrDie(
    Registry& registry, const Key& key, typename Registry::Function fn) {
  CHECK(registry.Register(key, std::move(fn)))
      << "Registration failed, see error log";
  return true;
}

template <typename Registry, typename Key = typename Registry::Key>
bool RegisterOrDie(
    Registry* registry, const Key& key, typename Registry::Function fn) {
  return RegisterOrDie(*registry, key, std::move(fn));
}

// RAII object for scoped registration.  Will die on duplicate key registration.
//
// See docs above for more details.
template <typename Registry>
class ScopedRegistration {
 public:
  ScopedRegistration(Registry& registry, const typename Registry::Key& key,
                     typename Registry::Function fn)
      : registry_(registry), key_(key) {
    RegisterOrDie(registry_.get(), key_.value(), std::move(fn));
  }
  ScopedRegistration(ScopedRegistration&& other) noexcept
      : registry_(other.registry_), key_(std::move(other.key_)) {
    other.key_ = std::nullopt;
  }
  ScopedRegistration& operator=(ScopedRegistration&& other) noexcept {
    registry_.get().Unregister(key_.value());
    registry_ = other.registry_;
    key_ = std::move(other.key_);
    other.key_ = std::nullopt;
    return *this;
  }
  ~ScopedRegistration() {
    if (key_.has_value()) {
      registry_.get().Unregister(key_.value());
    }
  }

 private:
  std::reference_wrapper<Registry> registry_;
  std::optional<typename Registry::Key> key_;
};

// Deduction guide required to ensure class template argument deduction is
// correct.
template <typename Registry>
ScopedRegistration(Registry& registry, const typename Registry::Key& key,
                   typename Registry::Function fn)
    -> ScopedRegistration<Registry>;

}  // namespace profiler
}  // namespace tensorflow

#endif  // UTIL_REGISTRATION_FUNCTION_REGISTRY_H_
