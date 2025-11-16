import {Action, ActionReducer, createReducer, on} from '@ngrx/store';

import * as actions from './actions';
import {AppState, INIT_APP_STATE} from './state';
import {ActionCreatorAny} from './types';

/** Reduce functions of app stats. */
export const reducer: ActionReducer<AppState, Action> = createReducer(
    INIT_APP_STATE,
    on(
        actions.setActiveHeapObjectAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            memoryViewerState: {
              ...state.memoryViewerState,
              activeHeapObject: action.activeHeapObject,
            }
          };
        },
        ),
    on(
        actions.updateSelectedOpNodeChainAction,
        (state: AppState, action: ActionCreatorAny) => {
          let opNodeChain = state.opProfileState.selectedOpNodeChain;
          if (['by_program', 'by_category'].includes(
                  action.selectedOpNodeName)) {
            opNodeChain = [];
          }
          return {
            ...state,
            opProfileState: {
              ...state.opProfileState,
              selectedOpNodeChain: [...opNodeChain, action.selectedOpNodeName],
            }
          };
        },
        ),
    on(
        actions.setActiveOpProfileNodeAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            opProfileState: {
              ...state.opProfileState,
              activeOpProfileNode: action.activeOpProfileNode,
            }
          };
        },
        ),
    on(
        actions.setOpProfileRootNodeAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            opProfileState: {
              ...state.opProfileState,
              rootNode: action.rootNode,
            }
          };
        },
        ),
    on(
        actions.setOpAnalysisScalingFactorAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            opAnalysisState: {
              ...state.opAnalysisState,
              applyScalingFactor: action.applyScalingFactor,
            }
          };
        },
        ),
    on(
        actions.setProfilingDeviceTypeAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            profilingGeneralState: {
              ...state.profilingGeneralState,
              deviceType: action.deviceType,
            }
          };
        },
        ),
    on(
        actions.setActivePodViewerInfoAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            podViewerState: {
              ...state.podViewerState,
              activePodViewerInfo: action.activePodViewerInfo,
            }
          };
        },
        ),
    on(
        actions.setCapturingProfileAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            capturingProfile: action.capturingProfile,
          };
        },
        ),
    on(
        actions.setLoadingStateAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            loadingState: action.loadingState,
          };
        },
        ),
    on(
        actions.setCurrentToolStateAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            currentTool: action.currentTool,
          };
        },
        ),
    on(
        actions.setToolsInfoStateAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            toolsInfoState: action.toolsInfoState,
          };
        },
        ),
    on(
        actions.setHostsStateAction,
        (state: AppState, action) => {
          return {
            ...state,
            hostsState: action.hostsState,
          };
        },
        ),
    on(
        actions.setTagsStateAction,
        (state: AppState, action) => {
          return {
            ...state,
            tagsState: action.tagsState,
          };
        },
        ),
    on(
        actions.setDataRequestStateAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            dataRequest: action.dataRequest,
          };
        },
        ),
    on(
        actions.clearExportAsCsvStateAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            exportAsCsv: '',
          };
        },
        ),
    on(
        actions.setExportAsCsvStateAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            exportAsCsv: action.exportAsCsv,
          };
        },
        ),
    on(
        actions.setErrorMessageStateAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            errorMessage: action.errorMessage,
          };
        },
        ),
    on(
        actions.setCurrentRunAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            currentRun: action.currentRun,
          };
        },
        ),
    on(
        actions.updateRunToolsMapAction,
        (state: AppState, action: ActionCreatorAny) => {
          const {run, tools} = action;
          const updatedRunToolsMap = state.runToolsMap;
          updatedRunToolsMap[run] = tools;
          return {
            ...state,
            runToolsMap: updatedRunToolsMap,
          };
        },
        ),
    on(
        actions.setRunToolsMapAction,
        (state: AppState, action: ActionCreatorAny) => {
          return {
            ...state,
            runToolsMap: action.runToolsMap,
          };
        },
        ));

/** Reducer */
export function rootReducer(state: AppState|undefined, action: Action) {
  return reducer(state, action);
}
