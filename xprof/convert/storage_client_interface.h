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

#ifndef THIRD_PARTY_XPROF_CONVERT_STORAGE_CLIENT_INTERFACE_H_
#define THIRD_PARTY_XPROF_CONVERT_STORAGE_CLIENT_INTERFACE_H_

#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xprof {
namespace internal {

// Interface for storage operations to decouple from heavy cloud headers
// and facilitate mocking in tests.
class StorageClientInterface {
 public:
  virtual ~StorageClientInterface() = default;

  // Returns the size of the object in bytes.
  virtual absl::StatusOr<std::uint64_t> GetObjectSize(
      const std::string& bucket, const std::string& object) = 0;

  // Reads a range of bytes from the object into the buffer.
  virtual absl::Status ReadObject(const std::string& bucket,
                                  const std::string& object,
                                  std::uint64_t start, std::uint64_t end,
                                  char* buffer) = 0;

  // Writes the entire contents to the object.
  virtual absl::Status WriteObject(const std::string& bucket,
                                   const std::string& object,
                                   const std::string& contents) = 0;
};

}  // namespace internal
}  // namespace xprof

#endif  // THIRD_PARTY_XPROF_CONVERT_STORAGE_CLIENT_INTERFACE_H_
