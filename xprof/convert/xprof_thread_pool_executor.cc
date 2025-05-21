#include "xprof/convert/xprof_thread_pool_executor.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "tsl/platform/cpu_info.h"

namespace tensorflow {
namespace profiler {

XprofThreadPoolExecutor::XprofThreadPoolExecutor(const std::string& name,
                                                 int num_threads) {
  int effective_num_threads = num_threads;
  if (effective_num_threads <= 0) {
    effective_num_threads = tsl::port::MaxParallelism();
  }
  LOG(INFO) << "Creating " << name << " XprofThreadPoolExecutor with "
            << effective_num_threads << " threads.";
  thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), name, effective_num_threads,
      /* low_latency_hint */ true);
}

XprofThreadPoolExecutor::~XprofThreadPoolExecutor() {
  // The ThreadPool destructor automatically joins all threads.
  // If JoinAll() hasn't been called explicitly, it happens here.
  // No explicit action needed unless JoinAll was intended to be mandatory
  // before destruction.
}

void XprofThreadPoolExecutor::Execute(std::function<void()> fn) {
  if (thread_pool_) {
    thread_pool_->Schedule(std::move(fn));
  } else {
    // This can happen if Execute is called after JoinAll.
    LOG(WARNING) << "Attempted to schedule task on an already joined "
                    "XProfThreadPoolExecutor.";
  }
}

void XprofThreadPoolExecutor::JoinAll() {
  // Destroying the thread pool waits for all scheduled tasks to complete.
  // Resetting the unique_ptr triggers the ThreadPool destructor.
  thread_pool_.reset();
}

}  // namespace profiler
}  // namespace tensorflow
