#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_ANIMATION_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_ANIMATION_H_

#include <algorithm>
#include <cmath>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_set.h"

namespace traceviewer {

// This enables type erasure, allowing heterogeneous collections of different
// animated<T> instantiations (e.g., animated<float>, animated<Im2Vec>)
// to be stored together in a single container like flat_hash_set<Animation*>.
// The Animation::UpdateAll function can then iterate over and update all
// active animations without needing to know their specific underlying type.
class Animation {
 public:
  Animation() = default;
  virtual ~Animation() = default;

  // Updates all registered animations by delta_time and calls on_finished
  // for those that have completed.
  static void UpdateAll(float delta_time) {
    absl::erase_if(*animations_, [&](Animation* anim) {
      if (anim->Update(delta_time)) {
        finished_->push_back(anim);
        return true;
      }
      return false;
    });

    for (auto* anim : *finished_) anim->on_finished();
    finished_->clear();
  }

  // Returns true if there are any active animations.
  static bool HasActiveAnimations() { return !animations_->empty(); }

 protected:
  virtual void on_finished() = 0;

  // Registers an animation to be updated by UpdateAll.
  static void Register(Animation* animation) {
    if (!animation) return;
    animations_->insert(animation);
  }

  // Unregisters an animation from being updated by UpdateAll.
  static void Unregister(Animation* animation) {
    if (!animation) return;
    animations_->erase(animation);
  }

  // Updates the animation by delta_time. Returns true if finished.
  virtual bool Update(float delta_time) = 0;

 private:
  // Set of all animations that are currently active and being updated.
  inline static absl::NoDestructor<absl::flat_hash_set<Animation*>> animations_;
  // Buffer for animations that finished in the current UpdateAll call.
  inline static absl::NoDestructor<std::vector<Animation*>> finished_;
};

// Animated<T> represents a value of type T that can be animated over time
// towards a target value.
// Type T must support copy/move construction, assignment, operators `==`, `!=`,
// `+`, `-`, `*` with float, and `abs(T)` must be defined and return a type
// comparable with float.
template <typename T>
class Animated : public Animation {
  static_assert(std::is_trivially_destructible_v<T>,
                "Animated<T> requires T to be trivially destructible.");

 public:
  using OnFinished = std::function<void(T)>;

  explicit Animated(
      T value = T(), T target = T(), OnFinished on_finished = [](const T&) {})
      : current_(value), target_(target), on_finished_(on_finished) {
    if (current_ != target_) Animation::Register(this);
  }

  // Copy constructor: deliberately doesn't register copied instance for
  // animation. The copied instance takes on other.current_ as its value and
  // target.
  Animated(const Animated& other)
      : current_(other.current_),
        target_(other.current_),
        on_finished_(other.on_finished_) {}

  Animated(Animated&& other) noexcept
      : current_(other.current_),
        target_(other.target_),
        on_finished_(std::move(other.on_finished_)) {
    Animation::Unregister(&other);
    if (current_ != target_) Animation::Register(this);
  }
  ~Animated() override { Animation::Unregister(this); }

  void set_on_finished(OnFinished on_finished) {
    on_finished_ = on_finished ? std::move(on_finished) : [](const T&) {};
  }

  // Sets a new target value and starts animating towards it.
  // The on_finished callback is reset to a no-op.
  Animated& operator=(const T& new_value) {
    if (target_ == new_value) return *this;
    target_ = new_value;
    on_finished_ = [](const T&) {};
    if (current_ != target_) {
      Animation::Register(this);
    }
    return *this;
  }

  // Sets a new target value and an on_finished callback, and starts
  // animating towards it.
  Animated& operator()(const T& new_value, OnFinished on_finished) {
    if (target_ == new_value) return *this;
    target_ = new_value;
    on_finished_ = on_finished ? std::move(on_finished) : [](const T&) {};
    if (current_ != target_) {
      Animation::Register(this);
    }
    return *this;
  }

  // Immediately sets the current and target values to new_value, stopping
  // any ongoing animation and unregistering this instance.
  void snap_to(const T& new_value) {
    Animation::Unregister(this);
    current_ = new_value;
    target_ = new_value;
    on_finished_ = [](const T&) {};
  }

  const T& target() const { return target_; }

  explicit operator T() const { return current_; }
  const T& operator*() const { return current_; }
  const T* operator->() const { return &current_; }

 protected:
  void on_finished() override { on_finished_(target_); }

  bool Converged() const {
    using std::abs;
    return abs(current_ - target_) <
           std::max(abs(target_) * kRelativeTolerance, kAbsoluteTolerance);
  }

  bool Update(float delta_time) override {
    float factor = std::min(kSpeed * delta_time, 1.0f);
    current_ = current_ * (1.0f - factor) + target_ * factor;
    if (Converged()) {
      current_ = target_;
      return true;
    }
    return false;
  }

 private:
  // The current value of the animation.
  T current_;
  // The target value of the animation.
  T target_;
  // Callback to invoke when the animation finishes.
  std::function<void(T)> on_finished_;
  // Controls the speed of the animation. Higher values result in faster
  // convergence towards the target value. The units are effectively s^-1,
  // as it's multiplied by delta_time (in seconds) to determine the
  // interpolation factor.
  static constexpr float kSpeed = 12.5f;
  // The relative tolerance is a fraction of the magnitude of the target value.
  // The absolute tolerance is a fixed value, independent of the target value.
  // The animation stops when the relative tolerance is exceeded, or both the
  // relative and absolute tolerances are exceeded.
  static constexpr double kRelativeTolerance = 1e-4;
  static constexpr double kAbsoluteTolerance = 1e-4;
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_ANIMATION_H_
