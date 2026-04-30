#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_PALETTES_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_PALETTES_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "frontend/app/components/trace_viewer_v2/color/colors.h"

namespace traceviewer {

inline const absl::flat_hash_map<std::string, ColorPalette::Preset>
    kPresetPalettes = {
        {"Material",
         {
             .background = 0xFF383226,        // dark_teal #263238
             .foreground = 0xFFFFFFEE,        // foreground #EEFFFF
             .midtone = 0xFF413B2C,           // highlight_color #2C3B41
             .flame_header = 0xFFFFAA82,      // blue #82AAFF
             .collapsed_header = 0xFF4F4737,  // line_number_color #37474F
             .expanded_header = 0xFF7A6E54,   // faded #546E7A
             .subtitle = 0xFFD6CCB2,          // paleblue #B2CCD6
             .ruler_text = 0xFFD6CCB2,        // paleblue
             .ruler_line = 0xFF7A6E54,        // faded
             .selection = 0xFF413B2C,         // highlight_color
             .trace_colors = {0xFFFFAA82, 0xFF8DE8C3, 0xFFEA92C7, 0xFF6BCBFF,
                              0xFF7053FF, 0xFFFFDD89, 0xFF7871F0},
             .flow_colors = {0xFFFFDD89, 0xFF6C8CF7, 0xFFEA92C7, 0xFF7053FF,
                             0xFF6BCBFF, 0xFF8DE8C3, 0xFFFFAA82},
         }},
        {"Dracula",
         {
             .background = 0xFF362A28,        // background #282a36
             .foreground = 0xFFF2F8F8,        // foreground #f8f8f2
             .midtone = 0xFF5A4744,           // selection #44475a
             .flame_header = 0xFFF993BD,      // purple #bd93f9
             .collapsed_header = 0xFF5A4744,  // selection
             .expanded_header = 0xFFA47262,   // comment #6272a4
             .subtitle = 0xFFFDE98B,          // cyan #8be9fd
             .ruler_text = 0xFF8CFAF1,        // yellow #f1fa8c
             .ruler_line = 0xFFA47262,        // comment
             .selection = 0xFF5A4744,         // selection
             .trace_colors = {0xFFD28B26, 0xFF7BFA50, 0xFFF993BD, 0xFF8CFAF1,
                              0xFF6CB8FF, 0xFFC679FF, 0xFFFDE98B},
             .flow_colors = {0xFFFDE98B, 0xFFC679FF, 0xFF8CFAF1, 0xFF6CB8FF,
                             0xFFF993BD, 0xFF7BFA50, 0xFFD28B26},
         }},
        {"Monokai",
         {
             .background = 0xFF222827,        // background_color #272822
             .foreground = 0xFFF2F8F8,        // Token #f8f8f2
             .midtone = 0xFF3E4849,           // highlight_color #49483e
             .flame_header = 0xFFEFD966,      // Keyword #66d9ef
             .collapsed_header = 0xFF3E4849,  // highlight_color
             .expanded_header = 0xFF779095,   // Comment #959077
             .subtitle = 0xFF2EE2A6,          // Name.Function #a6e22e
             .ruler_text = 0xFF74DBE6,        // String #e6db74
             .ruler_line = 0xFF779095,        // Comment
             .selection = 0xFF3E4849,         // highlight_color
             .trace_colors = {0xFFEFD966, 0xFF2EE2A6, 0xFF8946FF, 0xFFFF81AE},
             .flow_colors = {0xFF74DBE6, 0xFFFF81AE, 0xFF8946FF, 0xFF2EE2A6,
                             0xFFEFD966},
         }},
        {"Solarized Dark",
         {
             .background = 0xFF362B00,        // base03 #002b36
             .foreground = 0xFF969483,        // base0 #839496
             .midtone = 0xFF423607,           // base02 #073642
             .flame_header = 0xFFD28B26,      // blue #268bd2
             .collapsed_header = 0xFF423607,  // base02
             .expanded_header = 0xFF756E58,   // base01 #586e75
             .subtitle = 0xFF98A12A,          // cyan #2aa198
             .ruler_text = 0xFFA1A193,        // base1 #93a1a1
             .ruler_line = 0xFF756E58,        // base01
             .selection = 0xFF423607,         // base02
             .trace_colors = {0xFFD28B26, 0xFF009985, 0xFF0089B5, 0xFF164BCB,
                              0xFF2F32DC, 0xFF8236D3, 0xFFC4716C},
             .flow_colors = {0xFF98A12A, 0xFFC4716C, 0xFF8236D3, 0xFF2F32DC,
                             0xFF164BCB, 0xFF0089B5, 0xFF009985},
         }},
        {"Solarized Light",
         {
             .background = 0xFFE3F6FD,        // base3 #fdf6e3
             .foreground = 0xFF837B65,        // base00 #657b83
             .midtone = 0xFFD5E8EE,           // base2 #eee8d5
             .flame_header = 0xFFD28B26,      // blue #268bd2
             .collapsed_header = 0xFFD5E8EE,  // base2
             .expanded_header = 0xFFA1A193,   // base1 #93a1a1
             .subtitle = 0xFF98A12A,          // cyan #2aa198
             .ruler_text = 0xFF756E58,        // base01 #586e75
             .ruler_line = 0xFFA1A193,        // base1
             .selection = 0xFFD5E8EE,         // base2
             .trace_colors = {0xFFD28B26, 0xFF009985, 0xFF0089B5, 0xFF164BCB,
                              0xFF2F32DC, 0xFF8236D3, 0xFFC4716C},
             .flow_colors = {0xFF98A12A, 0xFFC4716C, 0xFF8236D3, 0xFF2F32DC,
                             0xFF164BCB, 0xFF0089B5, 0xFF009985},
         }},
};

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_COLOR_PALETTES_H_
