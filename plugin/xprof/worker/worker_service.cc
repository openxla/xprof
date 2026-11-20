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

#include "plugin/xprof/worker/worker_service.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // NOLINT(build/c++17)
#include <functional>
#include <memory>
#include <string>
#include <system_error>  // NOLINT(build/c++11)
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/no_destructor.h"
#include "absl/base/thread_annotations.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "tsl/platform/host_info.h"
#include "tsl/platform/path.h"
#include "xprof/convert/profile_processor_factory.h"
#include "xprof/convert/tool_options.h"
#include "plugin/xprof/worker/grpc_utils.h"

namespace xprof {
namespace profiler {
namespace {

constexpr char kWorkerLogdirEnvVar[] = "XPROF_WORKER_LOGDIR";
constexpr absl::string_view kGcsScheme = "gs://";

ABSL_CONST_INIT absl::Mutex allowed_root_dirs_mutex(absl::kConstInit);
absl::NoDestructor<absl::flat_hash_set<std::string>> allowed_root_dirs
    ABSL_GUARDED_BY(allowed_root_dirs_mutex);
size_t allowed_local_root_count ABSL_GUARDED_BY(allowed_root_dirs_mutex) = 0;

std::string GetWorkerLogdirEnv() {
  const char* env_logdir = std::getenv(kWorkerLogdirEnvVar);
  return env_logdir != nullptr ? std::string(env_logdir) : std::string();
}

bool ConsumeRemotePrefix(absl::string_view* path) {
  return absl::ConsumePrefix(path, kGcsScheme);
}

bool IsRemotePath(absl::string_view path) { return ConsumeRemotePrefix(&path); }

absl::string_view StripTrailingSlashes(absl::string_view path) {
  while (path.size() > 1 && absl::EndsWith(path, "/")) {
    path.remove_suffix(1);
  }
  return path;
}

bool HasPathTraversalSegment(absl::string_view path) {
  for (absl::string_view segment :
       absl::StrSplit(path, absl::ByAnyChar("/\\"))) {
    if (segment == "." || segment == "..") {
      return true;
    }
  }
  return false;
}

bool IsUncPath(absl::string_view path, const std::filesystem::path& fs_path) {
  if (absl::StartsWith(path, "\\\\") || absl::StartsWith(path, "//")) {
    return true;
  }
  const std::string root_name = fs_path.root_name().generic_string();
  return absl::StartsWith(root_name, "//");
}

bool IsEmptyOrFilesystemRoot(const std::filesystem::path& fs_path) {
  return fs_path.relative_path().empty();
}

bool IsValidRemoteRelativePath(absl::string_view remote_rest) {
  return !remote_rest.empty() && !absl::StartsWith(remote_rest, "/") &&
         !absl::StrContains(remote_rest, "//") &&
         !absl::StrContains(remote_rest, '\\');
}

bool IsValidGcsObjectPath(absl::string_view path) {
  if (!absl::ConsumePrefix(&path, kGcsScheme)) {
    return false;
  }
  if (!IsValidRemoteRelativePath(path)) {
    return false;
  }
  const std::pair<absl::string_view, absl::string_view> bucket_and_object =
      absl::StrSplit(path, absl::MaxSplits('/', 1));
  return !bucket_and_object.first.empty() && !bucket_and_object.second.empty();
}

absl::StatusOr<std::string> ResolveRealPath(absl::string_view path) {
  if (path.empty()) {
    return absl::InvalidArgumentError("Path must not be empty.");
  }
  std::error_code ec;
  std::filesystem::path canonical_path = std::filesystem::canonical(
      std::filesystem::path(path.begin(), path.end()), ec);
  if (ec) {
    return absl::NotFoundError(ec.message());
  }
  return canonical_path.generic_string();
}

bool IsNonEmptyRegularFile(absl::string_view canonical_path) {
  const std::filesystem::path fs_path(canonical_path.begin(),
                                      canonical_path.end());
  std::error_code ec;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(fs_path, ec);
  if (ec || !std::filesystem::is_regular_file(status)) {
    return false;
  }
  const std::uintmax_t size = std::filesystem::file_size(fs_path, ec);
  return !ec && size > 0;
}

std::string NormalizeRootDir(absl::string_view dir) {
  if (dir.empty() || absl::StrContains(dir, '\0') ||
      HasPathTraversalSegment(dir)) {
    return "";
  }
  absl::string_view raw_remote_rest = dir;
  if (ConsumeRemotePrefix(&raw_remote_rest)) {
    absl::string_view stripped_rest = StripTrailingSlashes(raw_remote_rest);
    if (!IsValidRemoteRelativePath(stripped_rest)) {
      return "";
    }
    return std::string(StripTrailingSlashes(dir));
  }
  const std::filesystem::path raw_path(dir.begin(), dir.end());
  if (IsUncPath(dir, raw_path)) {
    return "";
  }
  absl::StatusOr<std::string> real_root = ResolveRealPath(dir);
  if (!real_root.ok()) {
    return "";
  }
  const std::filesystem::path canonical_root(real_root->begin(),
                                             real_root->end());
  std::error_code ec;
  if (IsEmptyOrFilesystemRoot(canonical_root) ||
      !std::filesystem::is_directory(canonical_root, ec) || ec) {
    return "";
  }
  return std::string(StripTrailingSlashes(*real_root));
}

bool HasValidXplaneFilename(absl::string_view path) {
  absl::string_view basename = tsl::io::Basename(path);
  if (!absl::ConsumeSuffix(&basename, ".xplane.pb") &&
      !absl::ConsumeSuffix(&basename, ".xplane.riegeli")) {
    return false;
  }
  return !basename.empty() && !absl::StartsWith(basename, ".");
}

bool IsPathWithinRoot(absl::string_view path, absl::string_view root) {
  if (root == kGcsScheme) {
    return IsValidGcsObjectPath(path);
  }
  return path == root ||
         (absl::StartsWith(path, root) && path.size() > root.size() &&
          path[root.size()] == '/');
}

tensorflow::profiler::ToolOptions BuildToolOptions(
    const ::xprof::pywrap::WorkerProfileDataRequest& request) {
  tensorflow::profiler::ToolOptions tool_options;
  for (const auto& [key, value] : request.origin_request().parameters()) {
    tool_options[key] = value;
  }
  return tool_options;
}

std::string RegisterAndResolveStartupWorkerRootImpl(
    absl::string_view logdir, absl::string_view env_logdir) {
  if (!env_logdir.empty()) {
    return std::string(logdir);
  }
  return AddWorkerAllowedRootDir(logdir) ? "" : std::string(logdir);
}

}  // namespace

bool AddWorkerAllowedRootDir(absl::string_view root_dir) {
  std::string normalized = NormalizeRootDir(root_dir);
  if (normalized.empty()) return false;
  const bool is_local = !IsRemotePath(normalized);
  absl::MutexLock lock(allowed_root_dirs_mutex);
  if (allowed_root_dirs->insert(std::move(normalized)).second && is_local) {
    ++allowed_local_root_count;
  }
  return true;
}

std::string RegisterAndResolveStartupWorkerRoot(absl::string_view logdir) {
  return RegisterAndResolveStartupWorkerRootImpl(logdir, GetWorkerLogdirEnv());
}

namespace internal {

void ResetWorkerAllowedRootDirsForTesting() {
  absl::MutexLock lock(allowed_root_dirs_mutex);
  allowed_root_dirs->clear();
  allowed_local_root_count = 0;
}

std::unique_ptr<ProfileWorkerServiceImpl>
CreateProfileWorkerServiceWithEnvForTesting(int max_concurrent_requests,
                                            absl::string_view allowed_root_dir,
                                            absl::string_view env_logdir) {
  return std::unique_ptr<ProfileWorkerServiceImpl>(new ProfileWorkerServiceImpl(
      max_concurrent_requests,
      ProfileWorkerServiceImpl::ResolveInitialRootPolicy(allowed_root_dir,
                                                         env_logdir)));
}

std::string RegisterAndResolveStartupWorkerRootWithEnvForTesting(
    absl::string_view logdir, absl::string_view env_logdir) {
  return RegisterAndResolveStartupWorkerRootImpl(logdir, env_logdir);
}

}  // namespace internal

ProfileWorkerServiceImpl::ProfileWorkerServiceImpl(
    int max_concurrent_requests, absl::string_view allowed_root_dir)
    : ProfileWorkerServiceImpl(
          max_concurrent_requests,
          ResolveInitialRootPolicy(allowed_root_dir, GetWorkerLogdirEnv())) {}

::grpc::Status ProfileWorkerServiceImpl::GetProfileData(
    ::grpc::ServerContext* context,
    const ::xprof::pywrap::WorkerProfileDataRequest* request,
    ::xprof::pywrap::WorkerProfileDataResponse* response) {
  if (request == nullptr || response == nullptr) {
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                          "Request and response must not be null.");
  }
  const auto& origin_request = request->origin_request();

  absl::StatusOr<std::string> validated_session_path =
      ValidateSessionPath(origin_request.session_id());
  if (!validated_session_path.ok()) {
    LOG(ERROR) << "GetProfileData rejected session_id: "
               << validated_session_path.status();
    return ToGrpcStatus(validated_session_path.status());
  }

  LOG(INFO) << "GetProfileData tool:" << origin_request.tool_name()
            << " session:" << *validated_session_path << " params:{"
            << absl::StrJoin(origin_request.parameters(), ",",
                             absl::PairFormatter("="))
            << "}";

  const bool acquired_immediately = AcquireSemaphoreSlot(
      origin_request.tool_name(), *validated_session_path);
  auto cleanup = absl::MakeCleanup([this]() { semaphore_.Release(1); });
  if (!acquired_immediately) {
    validated_session_path = ValidateSessionPath(origin_request.session_id());
    if (!validated_session_path.ok()) {
      LOG(ERROR) << "GetProfileData rejected session_id after queue wait: "
                 << validated_session_path.status();
      return ToGrpcStatus(validated_session_path.status());
    }
  }
  absl::Time file_start_time = absl::Now();

  auto processor = xprof::ProfileProcessorFactory::GetInstance().Create(
      origin_request.tool_name(), BuildToolOptions(*request));
  if (!processor) {
    LOG(ERROR) << "GetProfileData failed: Can not find tool: "
               << origin_request.tool_name();
    return ::grpc::Status(::grpc::StatusCode::INVALID_ARGUMENT,
                          "Can not find tool: " + origin_request.tool_name());
  }

  absl::StatusOr<std::string> map_output_file =
      processor->Map(*validated_session_path);
  if (!map_output_file.ok()) {
    LOG(ERROR) << "GetProfileData failed with status: "
               << map_output_file.status();
    return ToGrpcStatus(map_output_file.status());
  }
  response->set_output(*map_output_file);
  response->set_worker_id(tsl::port::Hostname());

  LOG(INFO)
      << "ProfileWorkerServiceImpl::GetProfileData finished successfully for "
         "tool: "
      << origin_request.tool_name() << " session: " << *validated_session_path
      << " in " << absl::Now() - file_start_time
      << " by worker: " << response->worker_id();
  return ::grpc::Status::OK;
}

void ProfileWorkerServiceImpl::SetQueueWaitCallbackForTesting(
    std::function<void()> callback) {
  absl::MutexLock lock(queue_wait_callback_mutex_);
  on_queue_wait_for_testing_ = std::move(callback);
}

ProfileWorkerServiceImpl::ProfileWorkerServiceImpl(
    int max_concurrent_requests, InitialRootPolicy root_policy)
    : semaphore_(std::max(1, max_concurrent_requests)),
      root_access_mode_(root_policy.mode),
      instance_allowed_root_dir_(std::move(root_policy.root_dir)) {}

ProfileWorkerServiceImpl::InitialRootPolicy
ProfileWorkerServiceImpl::ResolveInitialRootPolicy(
    absl::string_view allowed_root_dir, absl::string_view env_logdir) {
  absl::string_view configured_root =
      allowed_root_dir.empty() ? env_logdir : allowed_root_dir;
  if (configured_root.empty()) {
    return {RootAccessMode::kUseGlobalAllowlist, ""};
  }
  if (configured_root == kGcsScheme) {
    return {RootAccessMode::kRestrictToInstanceRoot,
            std::string(configured_root)};
  }
  std::string normalized = NormalizeRootDir(configured_root);
  if (normalized.empty()) {
    return {RootAccessMode::kDenyAll, ""};
  }
  return {RootAccessMode::kRestrictToInstanceRoot, std::move(normalized)};
}

bool ProfileWorkerServiceImpl::HasAllowedLocalRoot() const {
  switch (root_access_mode_) {
    case RootAccessMode::kDenyAll:
      return false;
    case RootAccessMode::kRestrictToInstanceRoot:
      return !IsRemotePath(instance_allowed_root_dir_);
    case RootAccessMode::kUseGlobalAllowlist: {
      absl::MutexLock lock(allowed_root_dirs_mutex);
      return allowed_local_root_count > 0;
    }
  }
  return false;
}

bool ProfileWorkerServiceImpl::IsPathAllowed(absl::string_view candidate_path,
                                             bool is_remote) const {
  switch (root_access_mode_) {
    case RootAccessMode::kDenyAll:
      return false;
    case RootAccessMode::kRestrictToInstanceRoot:
      return IsRemotePath(instance_allowed_root_dir_) == is_remote &&
             IsPathWithinRoot(candidate_path, instance_allowed_root_dir_);
    case RootAccessMode::kUseGlobalAllowlist:
      break;
  }
  absl::MutexLock lock(allowed_root_dirs_mutex);
  absl::string_view prefix = StripTrailingSlashes(candidate_path);
  while (!prefix.empty() && prefix != "/") {
    if (IsRemotePath(prefix) == is_remote &&
        allowed_root_dirs->contains(prefix)) {
      return true;
    }
    const size_t slash_pos = prefix.rfind('/');
    if (slash_pos == absl::string_view::npos || slash_pos == 0) break;
    prefix = StripTrailingSlashes(prefix.substr(0, slash_pos));
  }
  return false;
}

absl::StatusOr<std::string> ProfileWorkerServiceImpl::ValidateRemoteSessionPath(
    absl::string_view session_id) const {
  absl::string_view remote_rest = session_id;
  ConsumeRemotePrefix(&remote_rest);
  if (!IsValidRemoteRelativePath(remote_rest) ||
      (absl::StartsWith(session_id, kGcsScheme) &&
       !IsValidGcsObjectPath(session_id))) {
    return absl::PermissionDeniedError(
        "Session path is outside the allowed root directory.");
  }
  if (!IsPathAllowed(session_id, /*is_remote=*/true)) {
    return absl::PermissionDeniedError(
        "Session path is outside the allowed root directory.");
  }
  return std::string(session_id);
}

absl::StatusOr<std::string> ProfileWorkerServiceImpl::ValidateLocalSessionPath(
    absl::string_view session_id) const {
  const std::filesystem::path fs_path(session_id.begin(), session_id.end());
  const std::filesystem::path parent_fs_path = fs_path.parent_path();
  // Uses std::filesystem::path rather than tsl::io::IsAbsolutePath, which only
  // checks path[0] == '/' and misses Windows drive paths (such as "C:/...").
  if (IsUncPath(session_id, fs_path) || !fs_path.is_absolute() ||
      tsl::io::CleanPath(session_id) != session_id ||
      absl::StrContains(session_id, "\\\\") ||
      IsEmptyOrFilesystemRoot(parent_fs_path)) {
    return absl::InvalidArgumentError(
        "Local session_id must be a clean absolute path.");
  }
  // Reject immediately before touching the local filesystem when no local root
  // is permitted (e.g., gs://-only workers, kDenyAll, or empty allowlist).
  if (!HasAllowedLocalRoot()) {
    return absl::PermissionDeniedError(
        "Session path is outside the allowed root directory.");
  }

  // Check parent_dir before resolving session_id so probing non-existent files
  // outside allowed roots fails with PERMISSION_DENIED rather than leaking file
  // existence via INVALID_ARGUMENT.
  std::string real_parent_dir =
      ResolveRealPath(parent_fs_path.generic_string()).value_or("");
  if (!IsPathAllowed(real_parent_dir, /*is_remote=*/false)) {
    return absl::PermissionDeniedError(
        "Session path is outside the allowed root directory.");
  }

  absl::StatusOr<std::string> real_session_path = ResolveRealPath(session_id);
  if (!real_session_path.ok()) {
    return absl::InvalidArgumentError("Session file does not exist.");
  }
  // Re-check authorization on the resolved target BEFORE inspecting its
  // filename or file metadata so symlinks escaping an allowed directory always
  // fail with PERMISSION_DENIED without leaking out-of-root target details.
  if (!IsPathAllowed(*real_session_path, /*is_remote=*/false)) {
    return absl::PermissionDeniedError(
        "Session file target is outside the allowed root directory.");
  }
  if (!HasValidXplaneFilename(*real_session_path)) {
    return absl::InvalidArgumentError(
        "Session file target must be a valid .xplane.pb or .xplane.riegeli "
        "file.");
  }
  if (!IsNonEmptyRegularFile(*real_session_path)) {
    return absl::InvalidArgumentError(
        "Session file must be an existing non-empty XPlane file.");
  }
  return *std::move(real_session_path);
}

absl::StatusOr<std::string> ProfileWorkerServiceImpl::ValidateSessionPath(
    absl::string_view session_id) const {
  if (absl::StrContains(session_id, '\0') ||
      !HasValidXplaneFilename(session_id)) {
    return absl::InvalidArgumentError(
        "session_id must be a valid .xplane.pb or .xplane.riegeli file.");
  }
  if (HasPathTraversalSegment(session_id)) {
    return absl::InvalidArgumentError(
        "Path traversal is not allowed in session_id.");
  }
  if (IsRemotePath(session_id)) {
    return ValidateRemoteSessionPath(session_id);
  }
  return ValidateLocalSessionPath(session_id);
}

bool ProfileWorkerServiceImpl::AcquireSemaphoreSlot(
    absl::string_view tool_name, absl::string_view session_id) {
  const bool acquired_immediately = semaphore_.TryAcquire(1);
  if (!acquired_immediately) {
    LOG(INFO) << "GetProfileData tool:" << tool_name
              << " session:" << session_id
              << " is waiting in queue as max concurrent "
              << "requests limit reached.";
    std::function<void()> queue_wait_callback;
    {
      absl::MutexLock lock(queue_wait_callback_mutex_);
      queue_wait_callback = on_queue_wait_for_testing_;
    }
    if (queue_wait_callback) {
      queue_wait_callback();
    }
    semaphore_.Acquire(1);
  }
  LOG(INFO) << "GetProfileData tool:" << tool_name << " session:" << session_id
            << " acquired the semaphore and is starting processing.";
  return acquired_immediately;
}

}  // namespace profiler
}  // namespace xprof
