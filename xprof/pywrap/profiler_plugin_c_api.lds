# Linker version script for profiler_plugin_c_api.so on Linux.
# Restricts dynamic exports (.dynsym) to the Python module init entry point
# and ctypes-bound C API symbols, hiding static/object dependency symbols
# (BoringSSL, Abseil, gRPC, Protobuf) even under Bazel --start-lib groups.
{
  global:
    # Python C extension init symbol.
    PyInit_profiler_plugin_c_api;

    # Core profiler plugin C API (profiler_plugin_c_api.h).
    FreeString;
    Trace;
    Monitor;
    StartContinuousProfiling;
    StopContinuousProfiling;
    GetSnapshot;
    XSpaceToToolsData;
    XSpaceToToolsDataFromByteString;
    StartGrpcServer;
    InitializeStubs;
    BuiltWithEmbedded;

    # Embedded LLO C API (conditionally compiled under EMBEDDED_FEATURES_ENABLED).
    # Bracket glob patterns match only the exact symbol names while avoiding
    # lld --no-undefined-version errors when enable_embedded_features is False.
    [C]reateLloAnalysis;
    [G]etTotalInstructions;
    [G]etUniqueRegisters;
    [G]etNumUniqueOpcodes;
    [G]etOpcodeAtIndex;
    [G]etOpcodeCountAtIndex;
    [G]etLloAnalysisJson;
    [G]etLloStaticAnalysisJson;
    [G]etLloDebugString;
    [F]reeLloAnalysis;

    # ProfileData C API (profile_data_c_api.h).
    profile_data_from_bytes;
    profile_data_free;
    profile_data_num_planes;
    profile_data_get_plane;
    profile_plane_name;
    profile_plane_num_lines;
    profile_plane_get_line;
    profile_plane_num_stats;
    profile_plane_get_stat;
    profile_plane_free;
    profile_line_name;
    profile_line_num_events;
    profile_line_get_event;
    profile_line_free;
    profile_event_name;
    profile_event_start_ns;
    profile_event_duration_ns;
    profile_event_num_stats;
    profile_event_get_stat;
    profile_event_free;

  local:
    *;
};
