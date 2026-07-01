#include "xprof/convert/source_info_utils.h"

#include "gtest/gtest.h"
#include "plugin/xprof/protobuf/source_info.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

TEST(SourceInfoUtilsTest, ValidSourceInfo) {
  SourceInfo source_info;
  source_info.set_file_name("foo.cc");
  source_info.set_line_number(42);
  source_info.set_stack_frame("frame1\nframe2");
  EXPECT_EQ(
      SourceInfoFormattedText(source_info),
      "<div class='source-info-cell' title='frame1\nframe2'>foo.cc:42</div>");
}

TEST(SourceInfoUtilsTest, EmptySourceInfo) {
  SourceInfo source_info;
  EXPECT_EQ(SourceInfoFormattedText(source_info), "");
}

TEST(SourceInfoUtilsTest, SourceInfoMissingLine) {
  SourceInfo source_info;
  source_info.set_file_name("foo.cc");
  source_info.set_line_number(-1);
  EXPECT_EQ(SourceInfoFormattedText(source_info), "");
}

TEST(SourceInfoUtilsTest, StackFrameWithHtml) {
  SourceInfo source_info;
  source_info.set_file_name("foo.cc");
  source_info.set_line_number(42);
  source_info.set_stack_frame("<embedded module '_launcher'>");
  EXPECT_EQ(SourceInfoFormattedText(source_info),
            "<div class='source-info-cell' title='&lt;embedded module "
            "&#39;_launcher&#39;&gt;'>foo.cc:42</div>");
}

TEST(SourceInfoUtilsTest, FileNameWithHtml) {
  SourceInfo source_info;
  source_info.set_file_name("a<b>c.cc");
  source_info.set_line_number(42);
  EXPECT_EQ(SourceInfoFormattedText(source_info),
            "<div class='source-info-cell' title=''>a&lt;b&gt;c.cc:42</div>");
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
