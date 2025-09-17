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

#ifndef XPROF_CONVERT_GRAPHVIZ_HELPER_H_
#define XPROF_CONVERT_GRAPHVIZ_HELPER_H_

#include <functional>
#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_graph_dumper.h"
#include "xla/tsl/platform/errors.h"

namespace tensorflow {
namespace profiler {

inline std::function<absl::StatusOr<std::string>(absl::string_view)>*
    url_renderer = nullptr;

// Convert dot into visual graph in html
inline std::string WrapDotInHtml(std::string dot,
                                 absl::string_view layout_engine = "dot") {
  return absl::StrReplaceAll(
      R"html(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style type="text/css">
    body {
      height: 100vh;
      margin: 0;
    }
    #graph-container {height:95vh;width:100%;padding:10px;display:block;}
    #graph-container svg { height: 100% !important; width: 100% !important;}
    .node, .cluster {cursor:pointer;}
    .cluster:hover, .node:hover {outline: solid 3px black;}
  </style>
</head>
<body>
  <script src="https://www.gstatic.com/external_hosted/hpcc_js_wasm/index.min.js"
      integrity="sha384-LigJPbR3TOfU/Xbb+PjiN1dGJYPweLk7kiGnaMgmxnUmKWaCFKbb5tH6iLlyVhPZ"
      crossorigin="anonymous"></script>
  <script src="https://www.gstatic.com/external_hosted/svg_pan_zoom/svg-pan-zoom.js"></script>
  <div id="graph-container"></div>
  <script>
    const cssregex = new RegExp('stylesheet=<([^]*)\n>\n', 'gm');
    const hpccWasm = window["@hpcc-js/wasm"];
    const data = `$DOT`;
    const results = cssregex.exec(data);
    // graphviz has problem dealing with large stylesheets.
    // https://github.com/tensorflow/tensorflow/issues/17220#issuecomment-369228492
    // In order to avoid the problem, remove the stylesheet from the dot and
    // insert it directly info the rendered SVG.

    let dot_data = data;
    let css_data = '';
    if (results !== null) {
        css_data = results[1].replace(/\s*data:.*\s*,/,''); // Strip content-type field.
        // CSS inside DOT is URL-escaped, so we must unescape it
        // before we can insert it into SVG.
        css_data = unescape(css_data);
        dot_data = data.replace(cssregex, ''); // Remove the stylesheet
    }

    var render_start = performance.now()
    function add_controls(svg) {
        var htmlblob = new Blob([document.documentElement.innerHTML],
                                {type: 'text/html'});
        var savehtml = document.createElement('a');
        savehtml.setAttribute('href', URL.createObjectURL(htmlblob));
        savehtml.setAttribute('download', 'graph.html');
        savehtml.innerHTML = " [Save HTML+SVG] ";
        document.body.append(savehtml);
        var svgblob = new Blob([svg.outerHTML], {type: 'image/svg'});
        var savesvg = document.createElement('a');
        savesvg.setAttribute('href', URL.createObjectURL(svgblob));
        savesvg.setAttribute('download', 'graph.svg');
        savesvg.innerHTML = " [Save SVG] ";
        document.body.append(savesvg);
        var dotblob =  new Blob([data], {type: 'text/dot'});
        var savedot = document.createElement('a');
        savedot.setAttribute('href', URL.createObjectURL(dotblob));
        savedot.setAttribute('download', 'graph.dot');
        savedot.innerHTML = " [Save DOT] ";
        document.body.append(savedot);
        // Will get called after embed element was loaded
        var render_end = performance.now();
        var render_note = document.createElement('div')
        render_note.innerHTML = 'Rendering took '
                                + (render_end - render_start).toFixed(2) + "ms."
        document.body.append(render_note);
    }
    const render_callback = svg => {
      const container = document.getElementById('graph-container')
      container.innerHTML = `${svg}<style>${css_data}</style>`;
      const panZoom = svgPanZoom(container.children[0], {
        zoomEnabled: true,
        dblClickZoomEnabled: false,
        controlIconsEnabled: true,
        maxZoom: 200,
        minZoom: 0,
      });
      add_controls(svg);
    };
    hpccWasm.graphviz.layout(dot_data, "svg", "$LAYOUT_ENGINE").then(render_callback);
  </script>
</body>
</html>
)html",
      {{"$DOT", dot}, {"$LAYOUT_ENGINE", layout_engine}});
}

// Precondition: (url_renderer != nullptr || format != kUrl).
//
// (We specify this as a precondition rather than checking it in here and
// returning an error because we want to fail quickly when there's no URL
// renderer available, and this function runs only after we've done all the work
// of producing dot for the graph.)
inline absl::Status CheckPrecondition(xla::RenderedGraphFormat format) {
  if (format == xla::RenderedGraphFormat::kUrl && url_renderer == nullptr) {
    return absl::FailedPreconditionError(
        "Can't render as URL; no URL renderer was registered.");
  }
  return absl::OkStatus();
}

// Convert dot into certain format
inline absl::StatusOr<std::string> WrapDotInFormat(
    std::string dot, xla::RenderedGraphFormat format) {
  TF_RETURN_IF_ERROR(CheckPrecondition(format));
  switch (format) {
    case xla::RenderedGraphFormat::kUrl:
      if (url_renderer == nullptr) {
        return absl::InternalError("url_renderer is null");
      }
      return (*url_renderer)(dot);
    case xla::RenderedGraphFormat::kHtml:
      return WrapDotInHtml(dot);
    case xla::RenderedGraphFormat::kDot:
      return std::string(dot);
  }
}

// Registers a function which implements RenderedGraphFormat::kUrl.
// The input to the function is dot, and the output should be a URL or an error.
// There can only be one active renderer, and the last call to this function
// wins.
inline void RegisterGraphvizURLRenderer(
    std::function<absl::StatusOr<std::string>(absl::string_view)> renderer) {
  if (url_renderer != nullptr) {
    LOG(WARNING) << "Multiple calls to RegisterGraphToURLRenderer. Last call "
                    "wins, but because order of initialization in C++ is "
                    "nondeterministic, this may not be what you want.";
  }
  delete url_renderer;
  url_renderer =
      new std::function<absl::StatusOr<std::string>(absl::string_view)>(
          std::move(renderer));
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // XPROF_CONVERT_GRAPHVIZ_HELPER_H_
