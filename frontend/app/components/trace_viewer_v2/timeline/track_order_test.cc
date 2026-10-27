#include "frontend/app/components/trace_viewer_v2/timeline/track_order.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace traceviewer {
namespace {

using ::testing::ElementsAre;
using DropPosition = ::traceviewer::TrackOrder::DropPosition;

// Returns the names of `tracks` in the display order of `order`.
std::vector<std::string> DisplayedNames(absl::Span<const TrackNode> tracks,
                                        const TrackOrder& order) {
  std::vector<std::string> names;
  for (const int index : order.DisplayOrder(tracks)) {
    names.emplace_back(tracks[index].name);
  }
  return names;
}

// Two processes with two threads each:
//   0 P1
//   1   t1
//   2   t2
//   3 P2
//   4   t1
//   5   t2
std::vector<TrackNode> TwoProcesses() {
  return {{1, "P1"}, {2, "t1"}, {2, "t2"}, {1, "P2"}, {2, "t1"}, {2, "t2"}};
}

TEST(ComputeTrackParentsTest, ReturnsClosestPrecedingLowerLevelTrack) {
  const std::vector<TrackNode> tracks = {{1, "P1"}, {2, "t1"}, {3, "s1"},
                                         {2, "t2"}, {1, "P2"}, {3, "deep"}};

  EXPECT_THAT(ComputeTrackParents(tracks), ElementsAre(-1, 0, 1, 0, -1, 4));
}

TEST(ComputeTrackParentsTest, EmptyTracks) {
  EXPECT_TRUE(ComputeTrackParents({}).empty());
}

TEST(TrackSubtreeEndTest, ReturnsEndOfSubtree) {
  const std::vector<TrackNode> tracks = TwoProcesses();

  EXPECT_EQ(TrackSubtreeEnd(tracks, 0), 3);
  EXPECT_EQ(TrackSubtreeEnd(tracks, 1), 2);
  EXPECT_EQ(TrackSubtreeEnd(tracks, 3), 6);
  EXPECT_EQ(TrackSubtreeEnd(tracks, 5), 6);
}

TEST(TrackSubtreeEndTest, ClampsOutOfRangeIndices) {
  const std::vector<TrackNode> tracks = TwoProcesses();

  EXPECT_EQ(TrackSubtreeEnd(tracks, -1), 0);
  EXPECT_EQ(TrackSubtreeEnd(tracks, 6), 6);
}

TEST(TrackOrderTest, DisplayOrderIsNaturalOrderByDefault) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  const TrackOrder order;

  EXPECT_TRUE(order.empty());
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(TrackOrderTest, MovingRootBeforeSiblingMovesItsSubtree) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;

  EXPECT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/0, DropPosition::kBefore));

  EXPECT_FALSE(order.empty());
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(3, 4, 5, 0, 1, 2));
}

TEST(TrackOrderTest, MovingAfterSiblingSkipsItsSubtree) {
  const std::vector<TrackNode> tracks = {
      {1, "A"}, {1, "B"}, {2, "b1"}, {2, "b2"}, {1, "C"}};
  TrackOrder order;

  EXPECT_TRUE(
      order.Move(tracks, /*source=*/0, /*target=*/1, DropPosition::kAfter));

  EXPECT_THAT(DisplayedNames(tracks, order),
              ElementsAre("B", "b1", "b2", "A", "C"));
}

TEST(TrackOrderTest, MovingChildOnlyAffectsItsParent) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;

  // Moves P2's t2 before P2's t1. P1's threads have the same names, but they
  // have a different parent.
  EXPECT_TRUE(
      order.Move(tracks, /*source=*/5, /*target=*/4, DropPosition::kBefore));

  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 1, 2, 3, 5, 4));
}

TEST(TrackOrderTest, SequentialMovesCompose) {
  const std::vector<TrackNode> tracks = {
      {1, "A"}, {1, "B"}, {1, "C"}, {1, "D"}};
  TrackOrder order;

  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/0, DropPosition::kBefore));
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/0, /*target=*/2, DropPosition::kAfter));

  EXPECT_THAT(DisplayedNames(tracks, order), ElementsAre("D", "B", "C", "A"));
}

TEST(TrackOrderTest, RejectsInvalidMoves) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;

  // Out of range.
  EXPECT_FALSE(order.Move(tracks, -1, 0, DropPosition::kBefore));
  EXPECT_FALSE(order.Move(tracks, 0, 6, DropPosition::kBefore));
  // Onto itself.
  EXPECT_FALSE(order.Move(tracks, 1, 1, DropPosition::kAfter));
  // Children of different parents.
  EXPECT_FALSE(order.Move(tracks, 1, 4, DropPosition::kBefore));
  // A process and a thread.
  EXPECT_FALSE(order.Move(tracks, 3, 1, DropPosition::kBefore));

  EXPECT_TRUE(order.empty());
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(TrackOrderTest, MoveToCurrentPositionIsNoOp) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;

  // t1 is already right before t2, and t2 right after t1.
  EXPECT_FALSE(
      order.Move(tracks, /*source=*/1, /*target=*/2, DropPosition::kBefore));
  EXPECT_FALSE(
      order.Move(tracks, /*source=*/2, /*target=*/1, DropPosition::kAfter));

  EXPECT_TRUE(order.empty());
}

TEST(TrackOrderTest, MovingBackToNaturalOrderClearsOrder) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;

  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/0, DropPosition::kBefore));
  ASSERT_FALSE(order.empty());
  // P1 (index 0) is displayed after P2's subtree; move it back to the front.
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/0, /*target=*/3, DropPosition::kBefore));

  EXPECT_TRUE(order.empty());
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(TrackOrderTest, ClearRestoresNaturalOrder) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;
  ASSERT_TRUE(order.Move(tracks, 3, 0, DropPosition::kBefore));

  order.Clear();

  EXPECT_TRUE(order.empty());
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(TrackOrderTest, OrderSurvivesReloadWithDifferentIndices) {
  TrackOrder order;
  const std::vector<TrackNode> tracks = TwoProcesses();
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/0, DropPosition::kBefore));
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/2, /*target=*/1, DropPosition::kBefore));

  // The reloaded data has an extra thread at the front of P1, which shifts
  // the indices of every other track.
  const std::vector<TrackNode> reloaded = {{1, "P1"}, {2, "t0"}, {2, "t1"},
                                           {2, "t2"}, {1, "P2"}, {2, "t1"},
                                           {2, "t2"}};

  EXPECT_THAT(DisplayedNames(reloaded, order),
              ElementsAre("P2", "t1", "t2", "P1", "t2", "t1", "t0"));
}

TEST(TrackOrderTest, NewTracksGoAfterOrderedTracksInNaturalOrder) {
  TrackOrder order;
  const std::vector<TrackNode> tracks = {{1, "A"}, {1, "B"}};
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/1, /*target=*/0, DropPosition::kBefore));

  const std::vector<TrackNode> reloaded = {
      {1, "X"}, {1, "A"}, {1, "Y"}, {1, "B"}};

  EXPECT_THAT(DisplayedNames(reloaded, order), ElementsAre("B", "A", "X", "Y"));
}

TEST(TrackOrderTest, AbsentTrackReturnsToItsPlace) {
  TrackOrder order;
  const std::vector<TrackNode> tracks = {{1, "A"}, {1, "B"}, {1, "C"}};
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/2, /*target=*/0, DropPosition::kBefore));
  ASSERT_THAT(DisplayedNames(tracks, order), ElementsAre("C", "A", "B"));

  // A disappears (for example, it has no events in the fetched range).
  const std::vector<TrackNode> without_a = {{1, "B"}, {1, "C"}};
  ASSERT_THAT(DisplayedNames(without_a, order), ElementsAre("C", "B"));
  // Moving B in the meantime keeps A's place in the stored order.
  ASSERT_TRUE(
      order.Move(without_a, /*source=*/0, /*target=*/1, DropPosition::kBefore));
  ASSERT_THAT(DisplayedNames(without_a, order), ElementsAre("B", "C"));

  EXPECT_THAT(DisplayedNames(tracks, order), ElementsAre("B", "C", "A"));
}

TEST(TrackOrderTest, MoveAcrossOnlyAbsentTracksIsNoOp) {
  TrackOrder order;
  const std::vector<TrackNode> tracks = {{1, "A"}, {1, "B"}, {1, "C"}};
  // Stored order: A, C, B.
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/2, /*target=*/1, DropPosition::kBefore));

  // Without C, B is already displayed right after A.
  const std::vector<TrackNode> without_c = {{1, "A"}, {1, "B"}};

  EXPECT_FALSE(
      order.Move(without_c, /*source=*/1, /*target=*/0, DropPosition::kAfter));
  EXPECT_THAT(DisplayedNames(tracks, order), ElementsAre("A", "C", "B"));
}

TEST(TrackOrderTest, MoveStartsFromDisplayedOrderWhenOrdinalsChanged) {
  TrackOrder order;
  // The stored order lists the second "w" thread first. In the reloaded data,
  // the first "w" thread is gone, so the remaining one has ordinal 0 and takes
  // the place of the stored second "w".
  ASSERT_TRUE(order.Deserialize(R"({"orders":[{"order":[["w",1],["m",0]]}]})"));
  const std::vector<TrackNode> tracks = {{1, "m"}, {1, "w"}};
  ASSERT_THAT(DisplayedNames(tracks, order), ElementsAre("w", "m"));

  EXPECT_FALSE(
      order.Move(tracks, /*source=*/0, /*target=*/1, DropPosition::kAfter));
  EXPECT_TRUE(
      order.Move(tracks, /*source=*/0, /*target=*/1, DropPosition::kBefore));

  EXPECT_THAT(DisplayedNames(tracks, order), ElementsAre("m", "w"));
}

TEST(TrackOrderTest, SiblingsWithTheSameNameAreDistinguished) {
  TrackOrder order;
  const std::vector<TrackNode> tracks = {
      {1, "P"}, {2, "worker"}, {2, "main"}, {2, "worker"}};

  // Moves the second "worker" thread to the front.
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/1, DropPosition::kBefore));

  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 3, 1, 2));
}

TEST(TrackOrderTest, ReordersGrandchildren) {
  TrackOrder order;
  const std::vector<TrackNode> tracks = {{1, "P"},  {2, "T"},  {3, "s1"},
                                         {3, "s2"}, {2, "U"},  {1, "Q"},
                                         {2, "T"},  {3, "s1"}, {3, "s2"}};

  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/2, DropPosition::kBefore));

  // Only the subtracks of P/T are reordered, not the ones of Q/T.
  EXPECT_THAT(order.DisplayOrder(tracks),
              ElementsAre(0, 1, 3, 2, 4, 5, 6, 7, 8));
}

TEST(TrackOrderTest, ReorderingParentKeepsChildrenOrder) {
  TrackOrder order;
  const std::vector<TrackNode> tracks = TwoProcesses();
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/2, /*target=*/1, DropPosition::kBefore));
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/0, DropPosition::kBefore));

  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(3, 4, 5, 0, 2, 1));
}

TEST(TrackOrderTest, NeverMovesTrackIntoAnotherSubtree) {
  TrackOrder order;
  const std::vector<TrackNode> tracks = {{1, "P"}, {2, "x"}, {2, "y"}};
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/2, /*target=*/1, DropPosition::kBefore));
  ASSERT_THAT(DisplayedNames(tracks, order), ElementsAre("P", "y", "x"));

  // In the reloaded data, x is nested deeper than its sibling y. Displaying y
  // first would turn x into a child of y, so x stays first.
  const std::vector<TrackNode> reloaded = {{1, "P"}, {3, "x"}, {2, "y"}};

  EXPECT_THAT(DisplayedNames(reloaded, order), ElementsAre("P", "x", "y"));
}

TEST(TrackOrderTest, SerializeReturnsEmptyStringWithoutCustomOrder) {
  const TrackOrder order;

  EXPECT_EQ(order.Serialize(), "");
}

TEST(TrackOrderTest, EmptyRoundTrip) {
  TrackOrder order;
  EXPECT_TRUE(order.empty());
  EXPECT_EQ(order.Serialize(), "");
  EXPECT_TRUE(order.Deserialize(""));
  EXPECT_TRUE(order.empty());
  EXPECT_TRUE(order.Deserialize("   "));
  EXPECT_TRUE(order.empty());
}

TEST(TrackOrderTest, DeserializeEmptyStringClearsOrders) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/0, DropPosition::kBefore));
  EXPECT_FALSE(order.empty());
  EXPECT_TRUE(order.Deserialize(""));
  EXPECT_TRUE(order.empty());
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(TrackOrderTest, SerializeAndDeserializeRoundTrip) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/0, DropPosition::kBefore));
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/2, /*target=*/1, DropPosition::kBefore));

  const std::string serialized = order.Serialize();
  EXPECT_EQ(serialized,
            R"({"orders":[)"
            R"({"parent":[],"order":[["P2",0],["P1",0]]},)"
            R"({"parent":[["P1",0]],"order":[["t2",0],["t1",0]]}]})");

  TrackOrder restored;
  ASSERT_TRUE(restored.Deserialize(serialized));
  EXPECT_FALSE(restored.empty());
  EXPECT_THAT(restored.DisplayOrder(tracks), ElementsAre(3, 4, 5, 0, 2, 1));
  EXPECT_EQ(restored.Serialize(), serialized);
}

TEST(TrackOrderTest, SerializeAndDeserializeSpecialCharactersAndOrdinals) {
  const std::string special_name =
      std::string("quote\"slash\\tab\tnewline\nctrl\x01\x1f") +
      "\xF0\x9F\x94\xA5";
  const std::vector<TrackNode> tracks = {
      {1, "P"}, {2, special_name}, {2, "other"}, {2, special_name}};
  TrackOrder order;
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/1, DropPosition::kBefore));

  const std::string expected_serialized =
      std::string(R"({"orders":[{"parent":[["P",0]],"order":)") +
      R"([["quote\"slash\\tab\tnewline\nctrl\u0001\u001f)" + "\xF0\x9F\x94\xA5" +
      R"(",1],["quote\"slash\\tab\tnewline\nctrl\u0001\u001f)" + "\xF0\x9F\x94\xA5" +
      R"(",0],["other",0]]}]})";
  EXPECT_EQ(order.Serialize(), expected_serialized);

  TrackOrder restored;
  ASSERT_TRUE(restored.Deserialize(order.Serialize()));
  EXPECT_THAT(restored.DisplayOrder(tracks), ElementsAre(0, 3, 1, 2));
}

TEST(TrackOrderTest, DeserializeAcceptsAnyMemberOrderAndUnicodeEscapes) {
  TrackOrder restored;
  ASSERT_TRUE(restored.Deserialize(
      R"( { "orders" : [ { "order" : [ [ "\u0050\u0032" , 0 ] , [ "P1" , 0 ] ] ,)"
      R"( "parent" : [ ] } ] } )"));

  const std::vector<TrackNode> tracks = TwoProcesses();
  EXPECT_THAT(restored.DisplayOrder(tracks), ElementsAre(3, 4, 5, 0, 1, 2));
}

TEST(TrackOrderTest, DeserializeUsesDefaultsForMissingMembers) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;

  // No "parent" (the root tracks) and no ordinals.
  ASSERT_TRUE(order.Deserialize(R"({"orders":[{"order":[["P2"],["P1"]]}]})"));
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(3, 4, 5, 0, 1, 2));

  // No "order": the entry is dropped.
  ASSERT_TRUE(order.Deserialize(R"({"orders":[{"parent":[]}]})"));
  EXPECT_TRUE(order.empty());

  // No "orders".
  ASSERT_TRUE(order.Deserialize("{}"));
  EXPECT_TRUE(order.empty());
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST(TrackOrderTest, DeserializeSkipsUnknownMembers) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;

  ASSERT_TRUE(order.Deserialize(
      R"({"version":2,"orders":[{"parent":[["P1",0]],"order":[["t2",0]],)"
      R"("pinned":{"nested":[1,true,null,-2.5e3,"x"]}}]})"));

  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 2, 1, 3, 4, 5));
}

TEST(TrackOrderTest, DeserializeKeepsRepeatedKeysInParentPath) {
  // A "python3" process with a "python3" thread, whose children are reordered.
  const std::vector<TrackNode> tracks = {
      {1, "python3"}, {2, "python3"}, {3, "a"}, {3, "b"}};
  const std::string serialized =
      R"({"orders":[)"
      R"({"parent":[["python3",0],["python3",0]],"order":[["b",0],["a",0]]}]})";
  TrackOrder order;

  ASSERT_TRUE(order.Deserialize(serialized));

  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 1, 3, 2));
  EXPECT_EQ(order.Serialize(), serialized);
}

TEST(TrackOrderTest, DeserializeDropsRepeatedKeys) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;

  ASSERT_TRUE(order.Deserialize(
      R"({"orders":[{"order":[["P2",0],["P2",0],["P1",0]]}]})"));

  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(3, 4, 5, 0, 1, 2));
  EXPECT_EQ(order.Serialize(),
            R"({"orders":[)"
            R"({"parent":[],"order":[["P2",0],["P1",0]]}]})");
}

TEST(TrackOrderTest, CrossSessionOrderAppliesWhenParentProcessNamesDiffer) {
  // Session A has process "/device:TPU:0 (host_a)" with child tracks
  // "Steps", "XLA Modules", "XLA Ops".
  const std::vector<TrackNode> session_a = {
      {1, "/device:TPU:0 (host_a)"},
      {2, "Steps"},
      {2, "XLA Modules"},
      {2, "XLA Ops"},
  };
  TrackOrder order;
  // User moves "XLA Ops" (index 3) before "Steps" (index 1).
  ASSERT_TRUE(
      order.Move(session_a, /*source=*/3, /*target=*/1, DropPosition::kBefore));
  ASSERT_THAT(
      DisplayedNames(session_a, order),
      ElementsAre("/device:TPU:0 (host_a)", "XLA Ops", "Steps", "XLA Modules"));

  // Session B is a different session where the device process has a different
  // name ("/device:GPU:0 (host_b)"), "XLA Modules" is missing, and a new track
  // "DMA" is present.
  const std::vector<TrackNode> session_b = {
      {1, "/device:GPU:0 (host_b)"},
      {2, "Steps"},
      {2, "XLA Ops"},
      {2, "DMA"},
  };
  EXPECT_THAT(DisplayedNames(session_b, order),
              ElementsAre("/device:GPU:0 (host_b)", "XLA Ops", "Steps", "DMA"));
}

TEST(TrackOrderTest, OrderAppliesToParentWithSameNamesAndOtherOrdinals) {
  // Two processes named "P". The children of the second one are reordered.
  const std::vector<TrackNode> tracks = {{1, "P"}, {2, "a"}, {2, "b"},
                                         {1, "P"}, {2, "a"}, {2, "b"}};
  TrackOrder order;
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/5, /*target=*/4, DropPosition::kBefore));

  // In the reloaded data, there is only one "P", so its ordinal is 0.
  const std::vector<TrackNode> reloaded = {{1, "P"}, {2, "a"}, {2, "b"}};

  EXPECT_THAT(DisplayedNames(reloaded, order), ElementsAre("P", "b", "a"));
}

TEST(TrackOrderTest, PresentParentsWithSameNameDoNotBorrowEachOthersOrder) {
  const std::vector<TrackNode> tracks = {
      {1, "P"}, {2, "a"}, {2, "b"},
      {1, "P"}, {2, "a"}, {2, "b"},
  };
  TrackOrder order;
  // Move children of second P: "b" (index 5) before "a" (index 4).
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/5, /*target=*/4, DropPosition::kBefore));
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 1, 2, 3, 5, 4));
}

TEST(TrackOrderTest, MoveBackToNaturalOrderOverridesFallback) {
  const std::vector<TrackNode> tracks = {
      {1, "P"}, {2, "a"}, {2, "b"},
  };
  TrackOrder order;
  // Stored order under a different ordinal [("P", 1)] -> [b, a].
  ASSERT_TRUE(order.Deserialize(
      R"({"orders":[{"parent":[["P",1]],"order":[["b",0],["a",0]]}]})"));
  // Initially, step 2 fallback applies to [("P", 0)] since [("P", 1)] is
  // absent.
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 2, 1));

  // Now move "a" (index 1 in tracks, displayed at position 2) before "b"
  // (index 2 in tracks).
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/1, /*target=*/2, DropPosition::kBefore));
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 1, 2));
}

TEST(TrackOrderTest, CrossSessionFallbackWhenReorderedProcessIsMissing) {
  const std::vector<TrackNode> session_a = {
      {1, "P1"}, {2, "common_thread"}, {2, "other"},
  };
  TrackOrder order;
  ASSERT_TRUE(
      order.Move(session_a, /*source=*/2, /*target=*/1, DropPosition::kBefore));

  // In session B, P1 is missing, but P2 with common_thread is present.
  const std::vector<TrackNode> session_b = {
      {1, "P2"}, {2, "common_thread"}, {2, "other"},
  };
  EXPECT_THAT(DisplayedNames(session_b, order),
              ElementsAre("P2", "other", "common_thread"));
}

TEST(TrackOrderTest, MoveWithDropPositionAfterOntoLastSibling) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;
  // Move P1 (index 0) after P2 (index 3). P2 is the last root sibling.
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/0, /*target=*/3, DropPosition::kAfter));
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(3, 4, 5, 0, 1, 2));
}

TEST(TrackOrderTest, DeserializeSurrogatePairAndRejectsLoneSurrogate) {
  TrackOrder order;
  // UTF-16 surrogate pair for U+1F525 (FIRE): \ud83d\udd25
  EXPECT_TRUE(order.Deserialize(
      R"({"orders":[{"order":[["\ud83d\udd25",0]]}]})"));
  EXPECT_FALSE(order.Deserialize(
      R"({"orders":[{"order":[["\ud83d",0]]}]})"));
  EXPECT_FALSE(order.Deserialize(
      R"({"orders":[{"order":[["\udd25",0]]}]})"));
}

TEST(TrackOrderTest, DeserializeRejectsUnknownFieldDeeperThanMaxDepth) {
  TrackOrder order;
  std::string deeply_nested = R"({"unknown":)";
  for (int i = 0; i < 35; ++i) deeply_nested.push_back('[');
  for (int i = 0; i < 35; ++i) deeply_nested.push_back(']');
  deeply_nested.push_back('}');
  EXPECT_FALSE(order.Deserialize(deeply_nested));
}

TEST(TrackOrderTest, DeserializeRejectsOrdinalOverflow) {
  TrackOrder order;
  EXPECT_FALSE(order.Deserialize(
      R"({"orders":[{"order":[["a",2147483648]]}]})"));
}

TEST(TrackOrderTest, HandlesTracksWithEmptyNamesWithoutError) {
  const std::vector<TrackNode> tracks = {
      {1, ""}, {2, "t1"}, {2, ""}, {2, "t2"}};
  TrackOrder order;
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/1, DropPosition::kBefore));
  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(0, 3, 1, 2));
}

TEST(TrackOrderTest, DeserializeRejectsMalformedJsonAndPreservesState) {
  const std::vector<TrackNode> tracks = TwoProcesses();
  TrackOrder order;
  ASSERT_TRUE(
      order.Move(tracks, /*source=*/3, /*target=*/0, DropPosition::kBefore));

  for (const absl::string_view json : {
           "{",
           "[]",
           "null",
           R"({"orders":{}})",
           R"({"orders":[{"order":["P1"]}]})",
           R"({"orders":[{"order":[["P1",-1]]}]})",
           R"({"orders":[{"order":[["P1",01]]}]})",
           R"({"orders":[{"order":[["P1",0,"extra"]]}]})",
           "{} trailing",
       }) {
    EXPECT_FALSE(order.Deserialize(json)) << json;
  }

  EXPECT_THAT(order.DisplayOrder(tracks), ElementsAre(3, 4, 5, 0, 1, 2));
}

}  // namespace
}  // namespace traceviewer
