/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/log/log.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpc/grpc.h"
#include "grpcpp/channel.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"
#include "plugin/xprof/protobuf/worker_service.grpc.pb.h"

namespace xprof {
namespace profiler {

using xprof::pywrap::grpc::XprofAnalysisWorkerService;

constexpr char kAddressDelimiter = ',';

// Service config for the gRPC channel. This config will be applied to all
// methods of the service. It enables a robust retry policy for transient errors
// (UNAVAILABLE, RESOURCE_EXHAUSTED, etc.), sets a 10-minute timeout, and
// configures client-side round-robin load balancing.
constexpr char kServiceConfigJson[] = R"pb(
    {
      "methodConfig":
      [ {
        "name":
        [ {}],
             "timeout": "600s",
             "retryPolicy": {
               "maxAttempts": 4,
               "initialBackoff": "2s",
               "maxBackoff": "120s",
               "backoffMultiplier": 2.0,
               "retryableStatusCodes": [
                 "UNAVAILABLE",
                 "RESOURCE_EXHAUSTED",
                 "INTERNAL",
                 "ABORTED",
                 "NOT_FOUND"
               ]
             }
      }],
      "loadBalancingConfig":
      [ { "round_robin": {} }]
    })pb";

ABSL_CONST_INIT absl::Mutex gStubsMutex(absl::kConstInit);
// gStubs holds the gRPC stubs for the worker services.
// It is a vector of unique_ptrs to ensure that the stubs are properly
// cleaned up when the program exits. absl::NoDestructor is used to prevent
// the vector from being destroyed during program shutdown.
//
// GetNextStub() returns a std::shared_ptr to a stub. This shared_ptr does
// not own the stub; ownership remains with the unique_ptr in the gStubs
// vector. A no-op deleter is provided to the shared_ptr to prevent it from
// attempting to delete the raw pointer. This allows multiple clients to
// safely share the stub without transferring ownership.
static absl::NoDestructor<
    std::vector<std::unique_ptr<XprofAnalysisWorkerService::Stub>>>
    gStubs ABSL_GUARDED_BY(gStubsMutex);
static std::atomic<size_t> gCurrentStubIndex = 0;
static std::atomic<bool> gStubsInitialized = false;

// Creates a gRPC channel for a given worker address. This channel is
// configured with a service config that enables a robust retry policy for
// transient errors and sets the client-side load balancing policy to
// round-robin.
static std::shared_ptr<::grpc::Channel> CreateWorkerChannelForAddress(
    absl::string_view address) {
  grpc::ChannelArguments args;
  args.SetServiceConfigJSON(kServiceConfigJson);
  args.SetLoadBalancingPolicyName("round_robin");
  args.SetInt(GRPC_ARG_DNS_MIN_TIME_BETWEEN_RESOLUTIONS_MS, 5000);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIME_MS, 20000);
  args.SetInt(GRPC_ARG_KEEPALIVE_TIMEOUT_MS, 10000);
  args.SetInt(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS, 1);
  args.SetInt(GRPC_ARG_ENABLE_RETRIES, 1);
  args.SetInt(GRPC_ARG_MAX_RECEIVE_MESSAGE_LENGTH, -1);
  args.SetInt(GRPC_ARG_MAX_SEND_MESSAGE_LENGTH, -1);

  // Create the channel with insecure credentials. This is acceptable because
  // the communication between the aggregator and workers happens within a
  // trusted, internal network environment.
  std::shared_ptr<::grpc::Channel> channel = ::grpc::CreateCustomChannel(
      std::string(address), ::grpc::InsecureChannelCredentials(), args);  // NOLINT
  LOG(INFO) << "Created gRPC channel for address: " << address;
  return channel;
}

void InitializeStubs(const std::string& worker_service_addresses) {
  absl::MutexLock lock(&gStubsMutex);
  if (gStubsInitialized.load(std::memory_order_acquire)) {
    // Already initialized.
    return;
  }
  std::vector<absl::string_view> addresses =
      absl::StrSplit(worker_service_addresses, kAddressDelimiter);
  for (absl::string_view address : addresses) {
    if (address.empty()) continue;
    std::shared_ptr<::grpc::Channel> channel =
        CreateWorkerChannelForAddress(address);
    gStubs->push_back(XprofAnalysisWorkerService::NewStub(channel));
  }
  gStubsInitialized.store(true, std::memory_order_release);
}

std::shared_ptr<XprofAnalysisWorkerService::Stub> GetNextStub() {
  absl::MutexLock lock(&gStubsMutex);
  if (!gStubsInitialized.load(std::memory_order_acquire) || gStubs->empty()) {
    return nullptr;
  }

  size_t index = gCurrentStubIndex.fetch_add(1, std::memory_order_acq_rel);
  // The returned shared_ptr does not own the stub. The stub's lifetime is
  // managed by the unique_ptrs in the gStubs vector. Thus, a no-op deleter is
  // provided to prevent the shared_ptr from attempting to delete the stub.
  return std::shared_ptr<XprofAnalysisWorkerService::Stub>(
      (*gStubs)[index % gStubs->size()].get(),
      [](XprofAnalysisWorkerService::Stub* ptr) { /*do nothing*/ });
}

}  // namespace profiler
}  // namespace xprof
