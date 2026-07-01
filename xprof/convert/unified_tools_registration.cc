/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xprof/convert/unified_tools_registration.h"

#include "xprof/convert/unified_hlo_stats_processor.h"
#include "xprof/convert/unified_memory_profile_processor.h"
#include "xprof/convert/unified_memory_viewer_processor.h"
#include "xprof/convert/unified_op_profile_processor.h"
#include "xprof/convert/unified_overview_page_processor.h"
#include "xprof/convert/unified_profile_processor_factory.h"
#include "xprof/convert/unified_utilization_viewer_processor.h"

namespace xprof {

void RegisterUnifiedToolRegistrations() {
  REGISTER_UNIFIED_PROFILE_PROCESSOR("hlo_stats", UnifiedHloStatsProcessor);
  REGISTER_UNIFIED_PROFILE_PROCESSOR("memory_profile",
                                     UnifiedMemoryProfileProcessor);
  REGISTER_UNIFIED_PROFILE_PROCESSOR("memory_viewer",
                                     UnifiedMemoryViewerProcessor);
  REGISTER_UNIFIED_PROFILE_PROCESSOR("op_profile", UnifiedOpProfileProcessor);
  REGISTER_UNIFIED_PROFILE_PROCESSOR("overview_page",
                                     UnifiedOverviewPageProcessor);
  REGISTER_UNIFIED_PROFILE_PROCESSOR("utilization_viewer",
                                     UnifiedUtilizationViewerProcessor);
}

}  // namespace xprof
