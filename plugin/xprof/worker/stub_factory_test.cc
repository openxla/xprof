#include "plugin/xprof/worker/stub_factory.h"

#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "<gtest/gtest.h>"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "grpcpp/server_context.h"
#include "grpcpp/support/status.h"
#include "plugin/xprof/protobuf/worker_service.grpc.pb.h"

namespace xprof {
namespace profiler {
namespace {

class MockProfileWorkerServiceImpl final
    : public ::xprof::pywrap::grpc::XprofAnalysisWorkerService::Service {
 public:
  ::grpc::Status GetProfileData(
      ::grpc::ServerContext* context,
      const ::xprof::pywrap::WorkerProfileDataRequest* request,
      ::xprof::pywrap::WorkerProfileDataResponse* response) override {
    if (++call_count_ <= 2) {
      return ::grpc::Status(::grpc::StatusCode::UNAVAILABLE, "Unavailable");
    }
    response->set_output("success");
    return ::grpc::Status::OK;
  }
  int call_count_ = 0;
};

class StubFactoryTest : public ::testing::Test {
 protected:
  void SetUp() override { internal::ResetStubsForTesting(); }
};

TEST_F(StubFactoryTest, RetryTest) {
  MockProfileWorkerServiceImpl service;
  ::grpc::ServerBuilder builder;
  int port;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(&service);
  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
  ASSERT_NE(server, nullptr);

  InitializeStubs("localhost:" + std::to_string(port));
  auto stub = GetNextStub();
  ASSERT_NE(stub, nullptr);

  ::grpc::ClientContext context;
  ::xprof::pywrap::WorkerProfileDataRequest request;
  ::xprof::pywrap::WorkerProfileDataResponse response;
  ::grpc::Status status = stub->GetProfileData(&context, request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(response.output(), "success");
  EXPECT_EQ(service.call_count_, 3);  // 2 failures + 1 success

  server->Shutdown();
  server->Wait();
}

class MockUnknownProfileWorkerServiceImpl final
    : public ::xprof::pywrap::grpc::XprofAnalysisWorkerService::Service {
 public:
  ::grpc::Status GetProfileData(
      ::grpc::ServerContext* context,
      const ::xprof::pywrap::WorkerProfileDataRequest* request,
      ::xprof::pywrap::WorkerProfileDataResponse* response) override {
    if (++call_count_ <= 2) {
      return ::grpc::Status(::grpc::StatusCode::UNKNOWN, "Unknown Error");
    }
    response->set_output("success");
    return ::grpc::Status::OK;
  }
  int call_count_ = 0;
};

TEST_F(StubFactoryTest, RetryTestUnknown) {
  MockUnknownProfileWorkerServiceImpl service;
  ::grpc::ServerBuilder builder;
  int port;
  builder.AddListeningPort("localhost:0", ::grpc::InsecureServerCredentials(),
                           &port);
  builder.RegisterService(&service);
  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
  ASSERT_NE(server, nullptr);

  InitializeStubs("localhost:" + std::to_string(port));
  auto stub = GetNextStub();
  ASSERT_NE(stub, nullptr);

  ::grpc::ClientContext context;
  ::xprof::pywrap::WorkerProfileDataRequest request;
  ::xprof::pywrap::WorkerProfileDataResponse response;
  ::grpc::Status status = stub->GetProfileData(&context, request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(service.call_count_, 3);

  server->Shutdown();
  server->Wait();
}

TEST_F(StubFactoryTest, NoStubs) { EXPECT_EQ(GetNextStub(), nullptr); }

TEST_F(StubFactoryTest, InitializeAndGetNextStub) {
  InitializeStubs("localhost:1234,localhost:5678");
  auto stub1 = GetNextStub();
  auto stub2 = GetNextStub();
  auto stub3 = GetNextStub();
  EXPECT_NE(stub1, nullptr);
  EXPECT_NE(stub2, nullptr);
  EXPECT_NE(stub3, nullptr);
  EXPECT_EQ(stub1, stub3);
}

TEST_F(StubFactoryTest, ConcurrentGetNextStub) {
  InitializeStubs("localhost:1000,localhost:2000,localhost:3000");
  constexpr int kNumThreads = 10;
  constexpr int kNumCallsPerThread = 100;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back([&]() {
      for (int j = 0; j < kNumCallsPerThread; ++j) {
        EXPECT_NE(GetNextStub(), nullptr);
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
}

TEST_F(StubFactoryTest, ConcurrentInitialize) {
  constexpr int kNumThreads = 10;
  std::vector<std::thread> threads;
  threads.reserve(kNumThreads);
  for (int i = 0; i < kNumThreads; ++i) {
    threads.emplace_back(
        [&]() { InitializeStubs("localhost:4000,localhost:5000"); });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  auto stub1 = GetNextStub();
  auto stub2 = GetNextStub();
  auto stub3 = GetNextStub();
  EXPECT_NE(stub1, nullptr);
  EXPECT_NE(stub2, nullptr);
  EXPECT_NE(stub3, nullptr);
  EXPECT_EQ(stub1, stub3);
}

TEST_F(StubFactoryTest, InitializeWithEmptyString) {
  InitializeStubs("");
  EXPECT_EQ(GetNextStub(), nullptr);
}

TEST_F(StubFactoryTest, InitializeWithMalformedString) {
  InitializeStubs("localhost:1111,,localhost:2222,");

  auto stub1 = GetNextStub();
  auto stub2 = GetNextStub();

  EXPECT_NE(stub1, nullptr);
  EXPECT_NE(stub2, nullptr);
  EXPECT_NE(stub1, stub2);
}

TEST_F(StubFactoryTest, ReinitializationIsIgnored) {
  InitializeStubs("localhost:1111");
  EXPECT_NE(GetNextStub(), nullptr);

  InitializeStubs("");

  EXPECT_NE(GetNextStub(), nullptr);
}

TEST_F(StubFactoryTest, ResetClearsStubs) {
  InitializeStubs("localhost:1234");
  EXPECT_NE(GetNextStub(), nullptr);

  internal::ResetStubsForTesting();

  EXPECT_EQ(GetNextStub(), nullptr);

  InitializeStubs("localhost:5678");
  EXPECT_NE(GetNextStub(), nullptr);
}

}  // namespace
}  // namespace profiler
}  // namespace xprof
