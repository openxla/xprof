#include "xprof/frontend/app/components/trace_viewer_v2/event_manager.h"

#include <emscripten/val.h>

#include <string>

#include "<gtest/gtest.h>"
#include "absl/strings/str_format.h"

namespace traceviewer {
namespace {

class EventManagerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    emscripten::val::global("window").call<void>(
        "eval", std::string("window.testResults = {};"));
  }

  void SetupEventListener(const std::string& event_name) {
    const std::string js_code = absl::StrFormat(
        R"(
          window.addEventListener('%s', (e) => {
            window.testResults['%s'] = { received: true, detail: e.detail };
          });
        )",
        event_name, event_name);
    emscripten::val::global("window").call<void>("eval",
                                                 emscripten::val(js_code));
  }
};

TEST_F(EventManagerTest, DispatchEvent) {
  const std::string event_name = "test-event";

  SetupEventListener(event_name);

  EventManager& event_manager = EventManager::Instance();
  EventData detail;
  detail["message"] = std::string("Hello from C++");
  detail["count"] = 1;

  event_manager.DispatchEvent(event_name, detail);

  // Check the results from JS.
  emscripten::val results =
      emscripten::val::global("window")["testResults"][event_name];

  ASSERT_TRUE(results["received"].as<bool>());

  emscripten::val event_detail = results["detail"];

  EXPECT_EQ(event_detail["message"].as<std::string>(), "Hello from C++");
  EXPECT_EQ(event_detail["count"].as<int>(), 1);
}

TEST_F(EventManagerTest, DispatchEventWithVariousTypes) {
  const std::string event_name = "various-types-event";

  SetupEventListener(event_name);

  EventManager& event_manager = EventManager::Instance();
  EventData detail;
  detail["float_val"] = 1.23f;
  detail["double_val"] = 4.56;
  detail["bool_val_true"] = true;
  detail["bool_val_false"] = false;

  event_manager.DispatchEvent(event_name, detail);

  // Check the results from JS.
  emscripten::val results =
      emscripten::val::global("window")["testResults"][event_name];

  ASSERT_TRUE(results["received"].as<bool>());

  emscripten::val event_detail = results["detail"];

  EXPECT_FLOAT_EQ(event_detail["float_val"].as<float>(), 1.23f);
  EXPECT_DOUBLE_EQ(event_detail["double_val"].as<double>(), 4.56);
  EXPECT_TRUE(event_detail["bool_val_true"].as<bool>());
  EXPECT_FALSE(event_detail["bool_val_false"].as<bool>());
}

TEST_F(EventManagerTest, DispatchEventWithNestedData) {
  const std::string event_name = "nested-event";

  SetupEventListener(event_name);

  EventManager& event_manager = EventManager::Instance();
  EventData detail;
  detail["level"] = 1;
  EventData nested_detail;
  nested_detail["value"] = std::string("nested");
  detail["nested"] = nested_detail;

  event_manager.DispatchEvent(event_name, detail);

  // Check the results from JS.
  emscripten::val results =
      emscripten::val::global("window")["testResults"][event_name];

  ASSERT_TRUE(results["received"].as<bool>());

  emscripten::val event_detail = results["detail"];

  EXPECT_EQ(event_detail["level"].as<int>(), 1);
  EXPECT_EQ(event_detail["nested"]["value"].as<std::string>(), "nested");
}

}  // namespace
}  // namespace traceviewer
