#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_FONTS_FONTS_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_FONTS_FONTS_H_
struct ImFont;

namespace traceviewer::fonts {

void LoadFonts(float pixel_ratio);

extern ImFont* body;  // Primary content text for paragraphs.

}  // namespace traceviewer::fonts

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_FONTS_FONTS_H_
