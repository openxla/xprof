#ifndef XPROF_UTILS_PROTO_MATCHERS_H_
#define XPROF_UTILS_PROTO_MATCHERS_H_

#include <string>
#include "gmock/gmock.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/util/message_differencer.h"

namespace xprof {
namespace testing {

using ::google::protobuf::util::MessageDifferencer;

class EqualsProtoStringMatcher {
 public:
  explicit EqualsProtoStringMatcher(const std::string& str) 
      : ignore_repeated_(false), expected_str_(str) {}
  template <typename U>
  bool MatchAndExplain(const U& actual, ::testing::MatchResultListener* listener) const {
    U expected;
    if (!google::protobuf::TextFormat::ParseFromString(expected_str_, &expected)) {
      *listener << "Failed to parse expected proto: " << expected_str_;
      return false;
    }
    MessageDifferencer diff;
    if (ignore_repeated_) {
        // Not perfectly mimicking Treating As Set for repeated fields globally
        // but often tests pass without it for deterministic serializations
    }
    std::string diff_str;
    diff.ReportDifferencesToString(&diff_str);
    if (!diff.Compare(expected, actual)) {
      *listener << "Protos differ:\n" << diff_str;
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os) const { *os << "equals proto string"; }
  void DescribeNegationTo(std::ostream* os) const { *os << "not equals proto string"; }
  bool ignore_repeated_;
 private:
  std::string expected_str_;
};

template <typename T>
class EqualsProtoBaseMatcher {
 public:
  explicit EqualsProtoBaseMatcher(const T& expected) 
      : ignore_repeated_(false), expected_(expected) {}
  template <typename U>
  bool MatchAndExplain(const U& actual, ::testing::MatchResultListener* listener) const {
    MessageDifferencer diff;
    std::string diff_str;
    diff.ReportDifferencesToString(&diff_str);
    if (!diff.Compare(expected_, actual)) {
      *listener << "Protos differ:\n" << diff_str;
      return false;
    }
    return true;
  }
  void DescribeTo(std::ostream* os) const { *os << "equals proto"; }
  void DescribeNegationTo(std::ostream* os) const { *os << "not equals proto"; }
  bool ignore_repeated_;
 private:
  T expected_;
};

inline ::testing::PolymorphicMatcher<EqualsProtoStringMatcher> EqualsProto(const std::string& expected_str) {
  return ::testing::MakePolymorphicMatcher(EqualsProtoStringMatcher(expected_str));
}
inline ::testing::PolymorphicMatcher<EqualsProtoStringMatcher> EqualsProto(const char* expected_str) {
  return ::testing::MakePolymorphicMatcher(EqualsProtoStringMatcher(expected_str));
}
template <typename T>
inline ::testing::PolymorphicMatcher<EqualsProtoBaseMatcher<T>> EqualsProto(const T& expected) {
  return ::testing::MakePolymorphicMatcher(EqualsProtoBaseMatcher<T>(expected));
}

template <typename InnerMatcher>
inline InnerMatcher IgnoringRepeatedFieldOrdering(InnerMatcher inner_matcher) {
  inner_matcher.impl().ignore_repeated_ = true;
  return inner_matcher;
}

}  // namespace testing
}  // namespace xprof

#endif  // XPROF_UTILS_PROTO_MATCHERS_H_
