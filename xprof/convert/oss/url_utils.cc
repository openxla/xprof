/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

namespace tensorflow {
namespace profiler {

// TODO(sannidhya): Implement this function as when required. This is a
// placeholder for now.
std::string Linkify(absl::string_view url, absl::string_view text) {
  return "";
}
// Generate a trace viewer URL. Used by both request and batch.
// TODO(sannidhya): Implement this function as when required. This is a
// placeholder for now.
std::string GenerateTraceViewerUrl(
    absl::string_view xprof_session_id, int64_t main_id, int32_t host_index,
    const google::protobuf::RepeatedField<int64_t>& related_ids) {
  return "";
}

}  // namespace profiler
}  // namespace tensorflow
