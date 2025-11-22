#ifndef PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_INPUT_HANDLER_H_
#define PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_INPUT_HANDLER_H_

#include <emscripten/html5.h>  // NOLINT

namespace traceviewer {

// The following functions are used as callbacks for Emscripten event handlers.
// They return EM_BOOL (typedef for int) as required by Emscripten:
// - true (1): The event was handled and should not be propagated further.
// - false (0): Allows the event to be processed by other listeners.

// Handles keydown events. Updates ImGui IO with the new key state and
// modifier keys. eventType and userData are unused.
EM_BOOL HandleKeyDown(int eventType, const EmscriptenKeyboardEvent* keyEvent,
                      void* userData);

// Handles keyup events. Updates ImGui IO with the new key state and
// modifier keys. eventType and userData are unused.
EM_BOOL HandleKeyUp(int eventType, const EmscriptenKeyboardEvent* keyEvent,
                    void* userData);

// Handles mouse move events.
EM_BOOL HandleMouseMove(int eventType, const EmscriptenMouseEvent* mouseEvent,
                        void* userData);

// Handles mouse down events.
EM_BOOL HandleMouseDown(int eventType, const EmscriptenMouseEvent* mouseEvent,
                        void* userData);

// Handles mouse up events.
EM_BOOL HandleMouseUp(int eventType, const EmscriptenMouseEvent* mouseEvent,
                      void* userData);

// Handles wheel events. Updates ImGui IO with wheel delta.
EM_BOOL HandleWheel(int eventType, const EmscriptenWheelEvent* wheelEvent,
                    void* userData);

}  // namespace traceviewer

#endif  // PERFTOOLS_ACCELERATORS_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_INPUT_HANDLER_H_
