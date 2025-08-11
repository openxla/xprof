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

#include "plugin/xprof/worker/stub_factory.h"

#include <atomic>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "plugin/xprof/protobuf/worker_service.grpc.pb.h"

namespace xprof {
namespace profiler {

using xprof::pywrap::grpc::XprofAnalysisWorkerService;

constexpr char kAddressDelimiter = ',';

ABSL_CONST_INIT absl::Mutex gStubsMutex(absl::kConstInit);
static std::vector<std::unique_ptr<XprofAnalysisWorkerService::Stub>>* gStubs
    ABSL_GUARDED_BY(gStubsMutex) = nullptr;
static std::atomic<size_t> gCurrentStubIndex = 0;

void InitializeStubs(const std::string& worker_service_addresses) {
  absl::MutexLock lock(&gStubsMutex);
  if (gStubs != nullptr) {
    // Already initialized.
    return;
  }
  gStubs = new std::vector<std::unique_ptr<XprofAnalysisWorkerService::Stub>>();
  std::vector<std::string> addresses =
      absl::StrSplit(worker_service_addresses, kAddressDelimiter);
  for (const std::string& address : addresses) {
    if (address.empty()) continue;
    std::shared_ptr<grpc::Channel> channel = grpc::CreateChannel(
        address, grpc::InsecureChannelCredentials());  // NOLINT
    gStubs->push_back(XprofAnalysisWorkerService::NewStub(channel));
  }
}

std::shared_ptr<XprofAnalysisWorkerService::Stub> GetNextStub() {
  absl::MutexLock lock(&gStubsMutex);
  if (gStubs == nullptr || gStubs->empty()) {
    return nullptr;
  }

  size_t index = gCurrentStubIndex.fetch_add(1, std::memory_order_relaxed);
  return std::shared_ptr<XprofAnalysisWorkerService::Stub>(
      (*gStubs)[index % gStubs->size()].get(),
      [](XprofAnalysisWorkerService::Stub* ptr) { /*do nothing*/ });
}

}  // namespace profiler
}  // namespace xprof
