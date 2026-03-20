#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_FONTS_FONTS_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_FONTS_FONTS_H_
struct ImFont;

namespace traceviewer::fonts {

void LoadFonts(float pixel_ratio);

extern ImFont* body_large;
extern ImFont* caption;
extern ImFont* label_large;
extern ImFont* label_medium;
extern ImFont* label_small;

}  // namespace traceviewer::fonts

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_FONTS_FONTS_H_
