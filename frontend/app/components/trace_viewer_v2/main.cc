#include <emscripten/bind.h>

#include "xprof/frontend/app/components/trace_viewer_v2/application.h"

int main(int argc, char** argv) {
  auto& app = traceviewer::Application::Instance();
  app.Initialize();
  app.Main();
  return 0;
}
