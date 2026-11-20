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

#ifndef XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_WORKER_SERVICE_H_
#define XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_WORKER_SERVICE_H_

#include <functional>
#include <memory>
#include <string>

#include "absl/base/thread_annotations.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "xla/pjrt/semaphore.h"
#include "plugin/xprof/protobuf/worker_service.grpc.pb.h"

namespace xprof {
namespace profiler {

class ProfileWorkerServiceImpl;

// Registers a root directory in the process-wide allowlist under which worker
// profile requests are permitted to read XPlane files and write cache
// artifacts. Returns true when `root_dir` normalizes to a valid non-empty root
// directory and is present in the allowlist; returns false for empty strings,
// invalid or nonexistent local paths, non-directory local files, bare remote
// schemes (such as "gs://"), paths with traversal segments, and filesystem
// roots (such as "/").
bool AddWorkerAllowedRootDir(absl::string_view root_dir);

// Resolves the per-instance `allowed_root_dir` argument for
// `InitializeGrpcServer` when starting the embedded gRPC worker for `logdir`.
// When `XPROF_WORKER_LOGDIR` is unset and `logdir` is valid, registers `logdir`
// in the process-wide allowlist and returns "" so subsequent dynamic session
// roots added via `AddWorkerAllowedRootDir` remain accessible; forwards
// `logdir` as the authoritative instance root without modifying the global
// allowlist when `XPROF_WORKER_LOGDIR` is set, when `logdir` is "gs://", or
// when `logdir` is rejected (failing closed).
std::string RegisterAndResolveStartupWorkerRoot(absl::string_view logdir);

namespace internal {
void ResetWorkerAllowedRootDirsForTesting();
std::unique_ptr<ProfileWorkerServiceImpl>
CreateProfileWorkerServiceWithEnvForTesting(int max_concurrent_requests,
                                            absl::string_view allowed_root_dir,
                                            absl::string_view env_logdir);
std::string RegisterAndResolveStartupWorkerRootWithEnvForTesting(
    absl::string_view logdir, absl::string_view env_logdir);
}  // namespace internal

class ProfileWorkerServiceImpl final
    : public ::xprof::pywrap::grpc::XprofAnalysisWorkerService::Service {
 public:
  // Initializes the worker service with the given concurrency limit (clamped to
  // at least 1) and an optional `allowed_root_dir` scoped to this service
  // instance. When `allowed_root_dir` (or `XPROF_WORKER_LOGDIR`) is
  // empty/unset, requests fall back to roots registered via
  // `AddWorkerAllowedRootDir`; when set to a non-empty value that normalizes to
  // empty (such as "/" or a nonexistent local directory), all requests fail
  // closed with `PERMISSION_DENIED`.
  explicit ProfileWorkerServiceImpl(int max_concurrent_requests,
                                    absl::string_view allowed_root_dir = "");

  ::grpc::Status GetProfileData(
      ::grpc::ServerContext* context,
      const ::xprof::pywrap::WorkerProfileDataRequest* request,
      ::xprof::pywrap::WorkerProfileDataResponse* response) override;

  // Installs a test callback invoked right before blocking on the concurrency
  // semaphore. Passing an empty function clears the callback.
  void SetQueueWaitCallbackForTesting(std::function<void()> callback);

 private:
  friend std::unique_ptr<ProfileWorkerServiceImpl>
  internal::CreateProfileWorkerServiceWithEnvForTesting(
      int max_concurrent_requests, absl::string_view allowed_root_dir,
      absl::string_view env_logdir);

  enum class RootAccessMode {
    kUseGlobalAllowlist,
    kRestrictToInstanceRoot,
    kDenyAll,
  };

  struct InitialRootPolicy {
    RootAccessMode mode;
    std::string root_dir;
  };

  ProfileWorkerServiceImpl(int max_concurrent_requests,
                           InitialRootPolicy root_policy);

  static InitialRootPolicy ResolveInitialRootPolicy(
      absl::string_view allowed_root_dir, absl::string_view env_logdir);
  bool HasAllowedLocalRoot() const;
  bool IsPathAllowed(absl::string_view candidate_path, bool is_remote) const;
  absl::StatusOr<std::string> ValidateRemoteSessionPath(
      absl::string_view session_id) const;
  absl::StatusOr<std::string> ValidateLocalSessionPath(
      absl::string_view session_id) const;
  absl::StatusOr<std::string> ValidateSessionPath(
      absl::string_view session_id) const;
  bool AcquireSemaphoreSlot(absl::string_view tool_name,
                            absl::string_view session_id);

  xla::Semaphore semaphore_;
  const RootAccessMode root_access_mode_;
  const std::string instance_allowed_root_dir_;
  mutable absl::Mutex queue_wait_callback_mutex_;
  std::function<void()> on_queue_wait_for_testing_
      ABSL_GUARDED_BY(queue_wait_callback_mutex_);
};
}  // namespace profiler
}  // namespace xprof

#endif  // XPROF_PLUGIN_TENSORBOARD_PLUGIN_PROFILE_WORKER_WORKER_SERVICE_H_
