#ifndef THIRD_PARTY_XPROF_PYWRAP_PROFILE_DATA_C_API_H_
#define THIRD_PARTY_XPROF_PYWRAP_PROFILE_DATA_C_API_H_

// C API wrapper for profile_data_lib.
//
// Provides a flat C ABI so Python can load this .so via ctypes without
// depending on any Python version. All functions use opaque handles.

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#define XPROF_EXPORT __declspec(dllexport)
#else
#define XPROF_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* ProfileData;
typedef void* ProfilePlane;
typedef void* ProfileLine;
typedef void* ProfileEvent;

XPROF_EXPORT ProfileData profile_data_from_bytes(const char* buf, size_t len);
XPROF_EXPORT void profile_data_free(ProfileData data);
XPROF_EXPORT int profile_data_num_planes(ProfileData data);
XPROF_EXPORT ProfilePlane profile_data_get_plane(ProfileData data, int index);

XPROF_EXPORT const char* profile_plane_name(ProfilePlane plane);
XPROF_EXPORT int profile_plane_num_lines(ProfilePlane plane);
XPROF_EXPORT ProfileLine profile_plane_get_line(ProfilePlane plane, int index);
XPROF_EXPORT int profile_plane_num_stats(ProfilePlane plane);
XPROF_EXPORT void profile_plane_get_stat(ProfilePlane plane, int index,
                                         const char** out_name,
                                         const char** out_value);
XPROF_EXPORT void profile_plane_free(ProfilePlane plane);

XPROF_EXPORT const char* profile_line_name(ProfileLine line);
XPROF_EXPORT int profile_line_num_events(ProfileLine line);
XPROF_EXPORT ProfileEvent profile_line_get_event(ProfileLine line, int index);
XPROF_EXPORT void profile_line_free(ProfileLine line);

XPROF_EXPORT const char* profile_event_name(ProfileEvent event);
XPROF_EXPORT double profile_event_start_ns(ProfileEvent event);
XPROF_EXPORT double profile_event_duration_ns(ProfileEvent event);
XPROF_EXPORT int profile_event_num_stats(ProfileEvent event);
XPROF_EXPORT void profile_event_get_stat(ProfileEvent event, int index,
                                         const char** out_name,
                                         const char** out_value);
XPROF_EXPORT void profile_event_free(ProfileEvent event);

#ifdef __cplusplus
}
#endif

#endif  // THIRD_PARTY_XPROF_PYWRAP_PROFILE_DATA_C_API_H_
