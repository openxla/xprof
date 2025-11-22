#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event_tree.h"

#include <vector>

#include "xprof/frontend/app/components/trace_viewer_v2/trace_helper/trace_event.h"
#include "<gtest/gtest.h>"
#include "absl/types/span.h"

namespace traceviewer {
namespace {

std::vector<const TraceEvent*> GetEventPointers(
    absl::Span<const TraceEvent> events) {
  std::vector<const TraceEvent*> events_ptr;
  events_ptr.reserve(events.size());
  for (const TraceEvent& event : events) {
    events_ptr.push_back(&event);
  }
  return events_ptr;
}

TEST(TraceEventTreeTest, EmptyEvents) {
  const std::vector<TraceEvent> events;
  auto events_ptr = GetEventPointers(events);
  const TraceEventTree tree = BuildTree(events_ptr);

  EXPECT_TRUE(tree.roots.empty());
  EXPECT_EQ(tree.max_depth, 0);
}

/**
 * |------------- Event -------------|
 */
TEST(TraceEventTreeTest, SingleEvent) {
  const std::vector<TraceEvent> events = {{.ts = 0, .dur = 10}};
  auto events_ptr = GetEventPointers(events);
  const TraceEventTree tree = BuildTree(events_ptr);

  ASSERT_EQ(tree.roots.size(), 1);
  EXPECT_EQ(tree.roots[0]->event->ts, 0);
  EXPECT_EQ(tree.roots[0]->event->dur, 10);
  EXPECT_EQ(tree.roots[0]->self_time, 10);
  EXPECT_EQ(tree.roots[0]->depth, 0);
  EXPECT_TRUE(tree.roots[0]->children.empty());
  EXPECT_EQ(tree.max_depth, 1);
}

/**
 * |----- Event A -----||----- Event B -----|
 */
TEST(TraceEventTreeTest, SequentialEvents) {
  const std::vector<TraceEvent> events = {{.ts = 0, .dur = 10},    // A
                                          {.ts = 10, .dur = 10}};  // B
  auto events_ptr = GetEventPointers(events);
  const TraceEventTree tree = BuildTree(events_ptr);

  ASSERT_EQ(tree.roots.size(), 2);
  EXPECT_EQ(tree.roots[0]->event->ts, 0);
  EXPECT_EQ(tree.roots[1]->event->ts, 10);
  EXPECT_EQ(tree.max_depth, 1);
}

/**
 * |---------- Event A ----------|
 *   |----- Event B -----|
 */
TEST(TraceEventTreeTest, NestedEvents) {
  const std::vector<TraceEvent> events = {{.ts = 0, .dur = 30},    // A
                                          {.ts = 10, .dur = 10}};  // B
  auto events_ptr = GetEventPointers(events);
  const TraceEventTree tree = BuildTree(events_ptr);

  ASSERT_EQ(tree.roots.size(), 1);
  EXPECT_EQ(tree.roots[0]->event->ts, 0);
  EXPECT_EQ(tree.roots[0]->self_time, 20);
  EXPECT_EQ(tree.roots[0]->depth, 0);
  ASSERT_EQ(tree.roots[0]->children.size(), 1);
  EXPECT_EQ(tree.roots[0]->children[0]->event->ts, 10);
  EXPECT_EQ(tree.roots[0]->children[0]->self_time, 10);
  EXPECT_EQ(tree.roots[0]->children[0]->depth, 1);
  EXPECT_EQ(tree.max_depth, 2);
}

/**
 * |---------- Event A ----------|
 *   |- B -|   |- C -|
 */
TEST(TraceEventTreeTest, MultipleChildren) {
  const std::vector<TraceEvent> events = {
      {.ts = 0, .dur = 30},  // A
      {.ts = 10, .dur = 5},  // B
      {.ts = 16, .dur = 5},  // C
  };
  auto events_ptr = GetEventPointers(events);
  const TraceEventTree tree = BuildTree(events_ptr);

  ASSERT_EQ(tree.roots.size(), 1);
  EXPECT_EQ(tree.roots[0]->event->ts, 0);
  EXPECT_EQ(tree.roots[0]->self_time, 20);  // 30 - (5 + 5)
  ASSERT_EQ(tree.roots[0]->children.size(), 2);
  EXPECT_EQ(tree.roots[0]->children[0]->event->ts, 10);
  EXPECT_EQ(tree.roots[0]->children[0]->self_time, 5);
  EXPECT_EQ(tree.roots[0]->children[0]->depth, 1);
  EXPECT_EQ(tree.roots[0]->children[1]->event->ts, 16);
  EXPECT_EQ(tree.roots[0]->children[1]->self_time, 5);
  EXPECT_EQ(tree.roots[0]->children[1]->depth, 1);
  EXPECT_EQ(tree.max_depth, 2);
}

/**
 * |---------- Event A ----------|
 *     |------- Event B -------|
 *       |----- Event C -----|
 */
TEST(TraceEventTreeTest, Grandchild) {
  const std::vector<TraceEvent> events = {
      {.ts = 0, .dur = 30},   // A
      {.ts = 5, .dur = 20},   // B
      {.ts = 10, .dur = 10},  // C
  };
  auto events_ptr = GetEventPointers(events);
  const TraceEventTree tree = BuildTree(events_ptr);

  ASSERT_EQ(tree.roots.size(), 1);
  EXPECT_EQ(tree.roots[0]->event->ts, 0);
  EXPECT_EQ(tree.roots[0]->self_time, 10);
  ASSERT_EQ(tree.roots[0]->children.size(), 1);
  EXPECT_EQ(tree.roots[0]->children[0]->event->ts, 5);
  EXPECT_EQ(tree.roots[0]->children[0]->self_time, 10);
  EXPECT_EQ(tree.roots[0]->children[0]->depth, 1);
  ASSERT_EQ(tree.roots[0]->children[0]->children.size(), 1);
  EXPECT_EQ(tree.roots[0]->children[0]->children[0]->event->ts, 10);
  EXPECT_EQ(tree.roots[0]->children[0]->children[0]->self_time, 10);
  EXPECT_EQ(tree.roots[0]->children[0]->children[0]->depth, 2);
  EXPECT_EQ(tree.max_depth, 3);
}

/**
 * |---------- Event A ----------|
 * |---------- Event B ----------|
 * |---------- Event C ----------|
 */
TEST(TraceEventTreeTest, EventsSameTime) {
  const std::vector<TraceEvent> events = {
      {.ts = 0, .dur = 10},  // A
      {.ts = 0, .dur = 10},  // B
      {.ts = 0, .dur = 10},  // C
  };
  auto events_ptr = GetEventPointers(events);
  const TraceEventTree tree = BuildTree(events_ptr);

  ASSERT_EQ(tree.roots.size(), 1);
  EXPECT_EQ(tree.max_depth, 3);

  const TraceEventNode* nodeA = tree.roots[0].get();

  EXPECT_EQ(nodeA->event->ts, 0);
  EXPECT_EQ(nodeA->event->dur, 10);
  EXPECT_EQ(nodeA->depth, 0);
  ASSERT_EQ(nodeA->children.size(), 1);

  const TraceEventNode* nodeB = nodeA->children[0].get();

  EXPECT_EQ(nodeB->event->ts, 0);
  EXPECT_EQ(nodeB->event->dur, 10);
  EXPECT_EQ(nodeB->depth, 1);
  ASSERT_EQ(nodeB->children.size(), 1);

  const TraceEventNode* nodeC = nodeB->children[0].get();

  EXPECT_EQ(nodeC->event->ts, 0);
  EXPECT_EQ(nodeC->event->dur, 10);
  EXPECT_EQ(nodeC->depth, 2);
  EXPECT_TRUE(nodeC->children.empty());

  // Self times
  EXPECT_EQ(nodeC->self_time, 10);
  EXPECT_EQ(nodeB->self_time, 0);
  EXPECT_EQ(nodeA->self_time, 0);
}

/**
 * |------- Event A -------|
 *           |------- Event B (ignored) -------|
 * Event B is not added as a child of A because it's not fully contained within
 * A.
 * Event B is also not added as a new root because A is still on the stack when
 * B is processed (B starts before A ends).
 */
TEST(TraceEventTreeTest, OverlappingEventNotContained) {
  const std::vector<TraceEvent> events = {{.ts = 0, .dur = 20},   // A
                                          {.ts = 1, .dur = 20}};  // B
  auto events_ptr = GetEventPointers(events);
  const TraceEventTree tree = BuildTree(events_ptr);

  ASSERT_EQ(tree.roots.size(), 1);
  EXPECT_EQ(tree.roots[0]->event->ts, 0);
  EXPECT_EQ(tree.roots[0]->self_time, 20);
  EXPECT_TRUE(tree.roots[0]->children.empty());
  EXPECT_EQ(tree.max_depth, 1);
}

/**
 * |------- Event A -------|    |- E -|
 *   |- B -|   |- D -|
 *    | C |
 */
TEST(TraceEventTreeTest, ComplexHierarchy) {
  const std::vector<TraceEvent> events = {
      {.ts = 0, .dur = 10},  // A
      {.ts = 1, .dur = 3},   // B
      {.ts = 2, .dur = 1},   // C
      {.ts = 5, .dur = 3},   // D
      {.ts = 11, .dur = 3},  // E
  };
  auto events_ptr = GetEventPointers(events);
  const TraceEventTree tree = BuildTree(events_ptr);

  ASSERT_EQ(tree.roots.size(), 2);
  EXPECT_EQ(tree.max_depth, 3);
  EXPECT_EQ(tree.roots[0]->event->ts, 0);
  EXPECT_EQ(tree.roots[0]->event->dur, 10);
  EXPECT_EQ(tree.roots[1]->event->ts, 11);
  EXPECT_EQ(tree.roots[1]->event->dur, 3);

  const TraceEventNode* nodeA = tree.roots[0].get();

  ASSERT_EQ(nodeA->children.size(), 2);

  const TraceEventNode* nodeB = nodeA->children[0].get();
  const TraceEventNode* nodeD = nodeA->children[1].get();

  EXPECT_EQ(nodeB->event->ts, 1);
  EXPECT_EQ(nodeD->event->ts, 5);

  ASSERT_EQ(nodeB->children.size(), 1);

  const TraceEventNode* nodeC = nodeB->children[0].get();

  EXPECT_EQ(nodeC->event->ts, 2);
  EXPECT_TRUE(nodeC->children.empty());
  EXPECT_TRUE(nodeD->children.empty());
}

/**
 * |------- Event A -------|    |- E -|
 * |- B -| |- D -|
 *   | C |
 */
TEST(TraceEventTreeTest, EventsStartEndTogether) {
  const std::vector<TraceEvent> events = {
      {.ts = 0, .dur = 10},  // A
      {.ts = 0, .dur = 3},   // B
      {.ts = 2, .dur = 1},   // C
      {.ts = 3, .dur = 3},   // D
      {.ts = 10, .dur = 3},  // E
  };
  auto events_ptr = GetEventPointers(events);
  const TraceEventTree tree = BuildTree(events_ptr);

  ASSERT_EQ(tree.roots.size(), 2);
  EXPECT_EQ(tree.max_depth, 3);
  EXPECT_EQ(tree.roots[0]->event->ts, 0);
  EXPECT_EQ(tree.roots[0]->event->dur, 10);
  EXPECT_EQ(tree.roots[1]->event->ts, 10);
  EXPECT_EQ(tree.roots[1]->event->dur, 3);

  const TraceEventNode* nodeA = tree.roots[0].get();

  ASSERT_EQ(nodeA->children.size(), 2);

  const TraceEventNode* nodeB = nodeA->children[0].get();
  const TraceEventNode* nodeD = nodeA->children[1].get();

  EXPECT_EQ(nodeB->event->ts, 0);
  EXPECT_EQ(nodeB->event->dur, 3);
  EXPECT_EQ(nodeD->event->ts, 3);
  EXPECT_EQ(nodeD->event->dur, 3);

  ASSERT_EQ(nodeB->children.size(), 1);

  const TraceEventNode* nodeC = nodeB->children[0].get();

  EXPECT_EQ(nodeC->event->ts, 2);
  EXPECT_EQ(nodeC->event->dur, 1);
  EXPECT_TRUE(nodeC->children.empty());
  EXPECT_TRUE(nodeD->children.empty());

  // Self times
  EXPECT_EQ(nodeC->self_time, 1);
  EXPECT_EQ(nodeB->self_time, 2);  // 3(B) - 1(C)
  EXPECT_EQ(nodeD->self_time, 3);
  EXPECT_EQ(nodeA->self_time, 4);  // 10(A) - 3(B) - 3(D)
  EXPECT_EQ(tree.roots[1]->self_time, 3);
}

}  // namespace
}  // namespace traceviewer
