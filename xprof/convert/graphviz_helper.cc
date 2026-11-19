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

#include "xprof/convert/graphviz_helper.h"

#include <cstddef>
#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"

namespace tensorflow {
namespace profiler {
namespace {

// Escapes a string for safe embedding inside a double-quoted JavaScript string
// literal within an HTML <script> block.
std::string JsStringEscape(absl::string_view raw) {
  std::string out;
  out.reserve(raw.size() + raw.size() / 8);
  for (size_t i = 0; i < raw.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(raw[i]);
    if (c < 0x20) {
      switch (c) {
        case '\b':
          out.append("\\b");
          break;
        case '\f':
          out.append("\\f");
          break;
        case '\n':
          out.append("\\n");
          break;
        case '\r':
          out.append("\\r");
          break;
        case '\t':
          out.append("\\t");
          break;
        default:
          absl::StrAppend(&out, "\\u", absl::Hex(c, absl::kZeroPad4));
          break;
      }
      continue;
    }
    switch (c) {
      case '"':
        out.append("\\\"");
        break;
      case '\\':
        out.append("\\\\");
        break;
      case '<':
        out.append("\\u003c");
        break;
      case '>':
        out.append("\\u003e");
        break;
      case '&':
        out.append("\\u0026");
        break;
      case 0xe2:
        if (i + 2 < raw.size() &&
            static_cast<unsigned char>(raw[i + 1]) == 0x80) {
          const unsigned char c2 = static_cast<unsigned char>(raw[i + 2]);
          if (c2 == 0xa8) {
            out.append("\\u2028");
            i += 2;
            break;
          }
          if (c2 == 0xa9) {
            out.append("\\u2029");
            i += 2;
            break;
          }
        }
        out.push_back(static_cast<char>(c));
        break;
      default:
        out.push_back(static_cast<char>(c));
        break;
    }
  }
  return out;
}

}  // namespace

std::string WrapDotInHtml(absl::string_view dot,
                          absl::string_view layout_engine) {
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
    const data = "$DOT";
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
        var svgblob = new Blob([svg], {type: 'image/svg'});
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
      {{"$DOT", JsStringEscape(dot)},
       {"$LAYOUT_ENGINE", JsStringEscape(layout_engine)}});
}

}  // namespace profiler
}  // namespace tensorflow
