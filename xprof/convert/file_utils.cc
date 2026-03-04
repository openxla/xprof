/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "xprof/convert/file_utils.h"

#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/protobuf.h"

namespace xprof {

absl::Status ReadBinaryProto(absl::string_view fname,
                             tsl::protobuf::MessageLite* proto) {
  std::string contents;
  const absl::Time start_download = absl::Now();
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(),
                                           std::string(fname), &contents));
  const absl::Time end_download = absl::Now();
  LOG(INFO) << "Download from GCS took: " << end_download - start_download;

  const absl::Time start_parse = absl::Now();
  if (!proto->ParseFromString(contents)) {
    return absl::DataLossError(
        absl::StrCat("Can't parse ", fname, " as binary proto"));
  }
  const absl::Time end_parse = absl::Now();
  LOG(INFO) << "Protobuf parsing took: " << end_parse - start_parse;
  LOG(INFO) << "File name" << fname;

  return absl::OkStatus();
}

absl::Status WriteBinaryProto(absl::string_view fname,
                              const tsl::protobuf::MessageLite& proto) {
  std::string contents;
  const absl::Time start_serialize = absl::Now();
  if (!proto.SerializeToString(&contents)) {
    return absl::InternalError(
        absl::StrCat("Failed to serialize proto to string for ", fname));
  }
  const absl::Time end_serialize = absl::Now();
  LOG(INFO) << "Proto serialization took: " << end_serialize - start_serialize;

  const absl::Time start_upload = absl::Now();
  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(tsl::Env::Default(),
                                            std::string(fname), contents));
  const absl::Time end_upload = absl::Now();
  LOG(INFO) << "Upload to GCS took: " << end_upload - start_upload;
  LOG(INFO) << "File name" << fname;
  return absl::OkStatus();
}

}  // namespace xprof
