#ifndef THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_INPUT_HANDLER_H_
#define THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_INPUT_HANDLER_H_

#include <emscripten/em_types.h>
#include <emscripten/html5.h>

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

// Handles mouse down events on the canvas. Also seeds the mouse position from
// the event, so that a click without a preceding mouse move (e.g. right after
// the mouse state was cleared on window blur) lands where the pointer is.
EM_BOOL HandleMouseDown(int eventType, const EmscriptenMouseEvent* mouseEvent,
                        void* userData);

// Handles mouse up events on the window, so that drags released outside the
// canvas still end. Releasing the last held button outside the canvas (or over
// an element covering it) invalidates the mouse position first, so that the
// release does not count as a click at the last in-canvas position.
EM_BOOL HandleMouseUp(int eventType, const EmscriptenMouseEvent* mouseEvent,
                      void* userData);

// Handles wheel events. Updates ImGui IO with wheel delta.
EM_BOOL HandleWheel(int eventType, const EmscriptenWheelEvent* wheelEvent,
                    void* userData);

// Handles window blur events. Clears the queued input events and the held
// mouse button, mouse position, modifier key, and shortcut key state when the
// browser window loses focus, without synthesizing releases (so no click or
// drop is committed on focus loss).
EM_BOOL HandleWindowBlur(int eventType, const EmscriptenFocusEvent* focusEvent,
                         void* userData);

}  // namespace traceviewer

#endif  // THIRD_PARTY_XPROF_FRONTEND_APP_COMPONENTS_TRACE_VIEWER_V2_INPUT_HANDLER_H_
