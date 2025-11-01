import {createFeatureSelector, createSelector,} from '@ngrx/store';

import {AppState, MemoryViewerState, OpProfileState, PodViewerState, STORE_KEY} from './state';
import {MemoizedSelectorAny} from './types';

const appState = createFeatureSelector<AppState>(STORE_KEY);

/** Selector for MemoryViewerState */
export const getMemoryViewerState: MemoizedSelectorAny = createSelector(
    appState, (appState: AppState) => appState.memoryViewerState);

/** Selector for ActiveHeapObjectState */
export const getActiveHeapObjectState: MemoizedSelectorAny = createSelector(
    getMemoryViewerState,
    (memoryViewerState: MemoryViewerState) =>
        memoryViewerState.activeHeapObject);

/** Selector for OpProfileState */
export const getOpProfileState: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.opProfileState);

/** Selector for OpAnalysisState */
export const getOpAnalysisState: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.opAnalysisState);

/** Selector for SelectedOpNodeChain */
export const getSelectedOpNodeChainState: MemoizedSelectorAny = createSelector(
    getOpProfileState,
    (opProfileState: OpProfileState) => opProfileState.selectedOpNodeChain);

/** Selector for ActiveOpProfileNodeState */
export const getActiveOpProfileNodeState: MemoizedSelectorAny = createSelector(
    getOpProfileState,
    (opProfileState: OpProfileState) => opProfileState.activeOpProfileNode);

/** Update Root node when loaded */
export const getOpProfileRootNode: MemoizedSelectorAny = createSelector(
    getOpProfileState,
    (opProfileState: OpProfileState) => opProfileState.rootNode);

/** Selector for getProfilingGeneralState */
export const getProfilingGeneralState: MemoizedSelectorAny = createSelector(
    appState, (appState: AppState) => appState.profilingGeneralState);

/** Selector for PodViewerState */
export const getPodViewerState: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.podViewerState);

/** Selector for ActivePodViewerInfoState */
export const getActivePodViewerInfoState: MemoizedSelectorAny = createSelector(
    getPodViewerState,
    (podViewerState: PodViewerState) => podViewerState.activePodViewerInfo);

/** Selector for CapturingProfileState */
export const getCapturingProfileState: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.capturingProfile);

/** Selector for LoadingState */
export const getLoadingState: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.loadingState);

/** Selector for CurrentTool */
export const getCurrentTool: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.currentTool);

/** Selector for ToolsInfoState */
export const getToolsInfoState: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.toolsInfoState);

/** Selector for hosts list */
export const getHostsState: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.hostsState);

/** Selector for tags list */
export const getTagsState: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.tagsState);

/** Selector for DataRequestType */
export const getDataRequest: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.dataRequest);

/** Selector for ExportAsCsv */
export const getExportAsCsv: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.exportAsCsv);

/** Selector for ErrorMessage */
export const getErrorMessage: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.errorMessage);

/** Selector for runTools */
export const getRunToolsMap: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.runToolsMap);

/** Selector for current run */
export const getCurrentRun: MemoizedSelectorAny =
    createSelector(appState, (appState: AppState) => appState.currentRun);
