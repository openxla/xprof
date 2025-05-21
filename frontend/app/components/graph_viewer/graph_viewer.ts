import {Component, ElementRef, inject, NgZone, OnDestroy, ViewChild} from '@angular/core';
import {MatSnackBar} from '@angular/material/snack-bar';
import {ActivatedRoute, Params, Router} from '@angular/router';
import {Store} from '@ngrx/store';
import {Throbber} from 'org_xprof/frontend/app/common/classes/throbber';
import {GRAPH_CENTER_NODE_COLOR, GRAPH_OP_COLORS} from 'org_xprof/frontend/app/common/constants/colors';
import {DIAGNOSTICS_DEFAULT, GRAPH_CONFIG_KEYS, GRAPH_TYPE_DEFAULT, GRAPHVIZ_PAN_ZOOM_CONTROL} from 'org_xprof/frontend/app/common/constants/constants';
import {OpProfileProto} from 'org_xprof/frontend/app/common/interfaces/data_table';
import {Diagnostics} from 'org_xprof/frontend/app/common/interfaces/diagnostics';
import {GraphConfigInput, GraphTypeObject, GraphViewerQueryParams} from 'org_xprof/frontend/app/common/interfaces/graph_viewer';
import * as utils from 'org_xprof/frontend/app/common/utils/utils';
import {GraphConfig} from 'org_xprof/frontend/app/components/graph_viewer/graph_config/graph_config';
import {OpProfileData} from 'org_xprof/frontend/app/components/op_profile/op_profile_data';
import {DATA_SERVICE_INTERFACE_TOKEN, DataServiceV2Interface} from 'org_xprof/frontend/app/services/data_service_v2/data_service_v2_interface';
import {setActiveOpProfileNodeAction, setCurrentToolStateAction, setOpProfileRootNodeAction, setProfilingDeviceTypeAction} from 'org_xprof/frontend/app/store/actions';
import {Node} from 'org_xprof/frontend/app/common/interfaces/op_profile.jsonpb_decls';
import {ReplaySubject} from 'rxjs';
import {takeUntil} from 'rxjs/operators';
import {locationReplace} from 'safevalues/dom';

const GRAPH_HTML_THRESHOLD = 1000000;  // bytes
const CENTER_NODE_GROUP_KEY = 'centerNode';

/** A graph viewer component. */
@Component({
  standalone: false,
  selector: 'graph-viewer',
  templateUrl: './graph_viewer.ng.html',
  styleUrls: ['./graph_viewer.scss'],
})
export class GraphViewer implements OnDestroy {
  readonly tool = 'graph_viewer';
  private readonly throbber = new Throbber(this.tool);
  private readonly dataService: DataServiceV2Interface =
      inject(DATA_SERVICE_INTERFACE_TOKEN);
  /** Handles on-destroy Subject, used to unsubscribe. */
  private readonly destroyed = new ReplaySubject<void>(1);

  @ViewChild(GraphConfig) config!: GraphConfig;
  @ViewChild('iframe', {static: false})
  graphRef!: ElementRef<HTMLIFrameElement>;

  sessionId = '';
  host = '';
  /** The hlo module list. */
  moduleList: string[] = [];
  initialParams: GraphConfigInput|undefined = undefined;
  selectedModule = '';
  opName = '';
  programId = '';
  graphWidth = 1;
  graphType = GRAPH_TYPE_DEFAULT;
  symbolId = '';
  symbolType = '';
  showMetadata = false;
  mergeFusion = false;
  opProfileLimit = 300;
  /** The graphviz url. */
  url = '';
  diagnostics: Diagnostics = {...DIAGNOSTICS_DEFAULT};
  graphvizUri = '';
  graphTypes: GraphTypeObject[] = [
    {label: 'Hlo Graph', value: GRAPH_TYPE_DEFAULT},
  ];
  loadingGraph = false;
  loadingModuleList = false;
  loadingOpProfile = false;
  loadingGraphvizUrl = false;
  opProfile: OpProfileProto|null = null;
  rootNode?: Node;
  data = new OpProfileData();
  selectedNode: Node|null = null;
  runtimeDataInjected = false;

  // ME related variables
  showMeGraph = false;

  constructor(
      public zone: NgZone,
      private readonly route: ActivatedRoute,
      private readonly store: Store<{}>,
      private readonly router: Router,
      private readonly snackBar: MatSnackBar,
  ) {
    this.route.params.pipe(takeUntil(this.destroyed)).subscribe((params) => {
      this.parseNavEvent(params);
      // init data that replis on the session id
      this.initData();
    });
    this.route.queryParams.pipe(takeUntil(this.destroyed))
        .subscribe((params) => {
          this.parseQueryParams(params);
          // Any graph viewer url query param change should trigger a potential
          // reload
          this.updateView();
        });
    this.store.dispatch(setCurrentToolStateAction({currentTool: this.tool}));
  }

  // process query params from in component navigation event
  parseQueryParams(params: Params) {
    this.showMeGraph = params['show_me_graph'] === 'true';
    // Plot the graph if node_name (op name) is provided in URL.
    this.opName = params['node_name'] || params['opName'] || '';
    this.selectedModule = params['module_name'] || '';
    this.programId = params['program_id'] || params['programId'] || '';
    this.graphWidth = Number(params['graph_width']) || 1;
    this.showMetadata = params['show_metadata'] === 'true';
    this.mergeFusion = params['merge_fusion'] === 'true';
    this.graphType =
        params['graph_type'] || params['graphType'] || GRAPH_TYPE_DEFAULT;
    this.symbolId = params['symbol_id'] || this.symbolId || '';
    this.symbolType = params['symbol_type'] || this.symbolType || '';
    this.opProfileLimit = params['op_profile_limit'] || 300;
  }

  // Process session id and host from sidenav navigation event
  parseNavEvent(params: Params) {
    this.sessionId =
        params['sessionId'] || params['run'] || this.sessionId || '';
    // host is a 3P only variable.
    this.host = params['host'] || this.host || '';
  }

  updateView() {
    // update graph_config input data
    this.initialParams = this.getParams();
    // refresh the graph view
    this.onPlot();
  }

  initData() {
    this.loadGraphTypes();
    this.loadModuleList();
    this.loadHloOpProfileData();
  }

  loadGraphTypes() {
    this.dataService.getGraphTypes(this.sessionId).subscribe((types) => {
      if (types) {
        this.graphTypes = types;
      }
    });
  }

  loadModuleList() {
    // Graph Viewer initial loading complete: module list loaded
    this.throbber.start();
    this.loadingModuleList = true;
    this.dataService.getModuleList(this.sessionId)
        .pipe(takeUntil(this.destroyed))
        .subscribe((moduleList) => {
          this.throbber.stop();
          if (moduleList) {
            this.moduleList = moduleList.split(',');
            if (!this.selectedModule) {
              // If moduleName not set in url, use default and try plot
              // again
              if (this.programId) {
                this.selectedModule =
                    this.moduleList.find(
                        (module: string) => module.includes(this.programId),
                        ) ||
                    this.moduleList[0];
              } else {
                this.selectedModule = this.moduleList[0];
                this.onPlot();
              }
            }
          }
          this.loadingModuleList = false;
        });
  }

  loadHloOpProfileData() {
    this.loadingOpProfile = true;
    const params = new Map<string, string>();
    params.set('op_profile_limit', this.opProfileLimit.toString());
    this.dataService.getOpProfileData(this.sessionId, this.host, params)
        .pipe(takeUntil(this.destroyed))
        .subscribe((data) => {
          if (data) {
            this.opProfile = data as OpProfileProto | null;
            if (this.opProfile) {
              this.store.dispatch(
                  setProfilingDeviceTypeAction({
                    deviceType: this.opProfile.deviceType,
                  }),
              );
            }
            // The root node will be ONLY used to calculate the TimeFraction
            // introduced in (CL/505580494) and this info will be used to
            // determine the FLOPS utilization of a node. However, unlike hlo op
            // profile, users can't speicify the root node. To have a consistent
            // result, use the default root node in hlo op profile.
            this.rootNode = this.opProfile!.byProgramExcludeIdle;
            this.store.dispatch(
                setOpProfileRootNodeAction({rootNode: this.rootNode}),
            );
          }
          this.loadingOpProfile = false;
          this.injectRuntimeData();
        });
  }

  installEventListeners() {
    const doc: Document|null = this.getGraphIframeDocument();
    if (!doc) return;

    const nodeElements = Array.from(doc.getElementsByClassName('node'));
    for (const e of nodeElements) {
      e.addEventListener('mouseenter', this.onHoverGraphvizNode.bind(this, e));
      e.addEventListener(
          'mouseleave',
          this.onHoverGraphvizNode.bind(this, null),
      );
      e.addEventListener(
          'dblclick', this.onDoubleClickGraphvizNode.bind(this, e));
      e.addEventListener(
          'click',
          this.onClickGraphvizNode.bind(this, e),
      );
    }
    const clusterElements = Array.from(doc.getElementsByClassName('cluster'));
    for (const e of clusterElements) {
      e.addEventListener(
          'mouseenter',
          this.onHoverGraphvizCluster.bind(this, e),
      );
      e.addEventListener(
          'mouseleave',
          this.onHoverGraphvizCluster.bind(this, null),
      );
      e.addEventListener(
          'dblclick', this.onDoubleClickGraphvizCluster.bind(this, e));
      e.addEventListener(
          'click',
          this.onClickGraphvizCluster.bind(this, e),
      );
    }
  }

  getGraphvizNodeOpName(element: HTMLElement|Element|null) {
    const opNameWithAvgTime =
        element?.getElementsByTagName('text')?.[0]?.textContent || '';
    // Split on space to remove the appended info (eg. avgTime)
    return opNameWithAvgTime.split(' ')[0];
  }

  getGraphvizClusterOpName(element: HTMLElement|Element|null) {
    const opNameWithAvgTime =
        element?.getElementsByTagName('text')?.[1]?.textContent || '';
    // Split on space to remove the appended info (eg. avgTime)
    return opNameWithAvgTime.split(' ')[0];
  }

  // Single click pin the op detail to the selected node
  onClickGraphvizNode(element: HTMLElement|Element, event: Event) {
    const opName = this.getGraphvizNodeOpName(element);
    this.selectedNode = this.getOpNodeInGraphviz(opName) || null;
    event.preventDefault();
  }

  onClickGraphvizCluster(element: HTMLElement|Element, event: Event) {
    const opName = this.getGraphvizClusterOpName(element);
    this.selectedNode = this.getOpNodeInGraphviz(opName) || null;
    event.preventDefault();
  }

  // Double click reload the graph centered on the selected node
  onDoubleClickGraphvizCluster(element: HTMLElement|Element, event: Event) {
    const opName = this.getGraphvizClusterOpName(element);
    this.onRecenterOpNode(opName);
  }

  onDoubleClickGraphvizNode(element: HTMLElement|Element, event: Event) {
    const opName = this.getGraphvizNodeOpName(element);
    this.onRecenterOpNode(opName);
  }

  onRecenterOpNode(opName: string) {
    // Don't re-navigate if click on the same center node
    if (this.opName === opName) return;
    this.zone.run(() => {
      this.opName = opName;
      this.onSearchGraph();
    });
  }

  // Hover display the op detail of the hovered node
  onHoverGraphvizNode(element: HTMLElement|Element|null) {
    // The node will display the op name in index 0 of "text" tag.
    const opName = this.getGraphvizNodeOpName(element);
    this.handleGraphvizHover(element, opName);
  }

  onHoverGraphvizCluster(element: HTMLElement|Element|null) {
    // The cluster will display the op name in index 1 of "text" tag.
    const opName = this.getGraphvizClusterOpName(element);
    this.handleGraphvizHover(element, opName);
  }

  updateAnchorOpNode = (node: Node|null) => {
    this.data.update(node || undefined);
    this.zone.run(() => {
      this.store.dispatch(
          setActiveOpProfileNodeAction({
            activeOpProfileNode: node || this.selectedNode || null,
          }),
      );
    });
  };

  handleGraphvizHover = (
      event: HTMLElement|Element|null,
      opName: string,
      ) => {
    if (!event) {
      this.updateAnchorOpNode(null);
      return;
    }
    if (opName) {
      const node = this.getOpNodeInGraphviz(opName);
      this.updateAnchorOpNode(node || null);
    }
  };

  getOpNodeInGraphviz(nodeName: string): Node|null|undefined {
    if (!this.opProfile || !this.rootNode) return null;
    for (const topLevelNode of this.rootNode.children!) {
      // Find the program id from HloOpProfile by the selected XLA module.
      // Assuming that the XLA modules and program ids are the same.
      if (topLevelNode.name === this.selectedModule) {
        const node = this.findNode(topLevelNode.children, nodeName);
        if (node) return node;
      }
    }
    return null;
  }

  findNode(
      children: Node[]|null|undefined,
      name: string,
      ): Node|null|undefined {
    if (!children) return null;
    for (const node of children) {
      if (node.name === name || node.name === `${name} and its duplicate(s)`) {
        return node;
      }
      const findChildren = this.findNode(node.children, name);
      if (findChildren) return findChildren;
    }
    return null;
  }

  private openSnackBar(message: string) {
    this.snackBar.open(message, 'Close.', {duration: 5000});
  }

  getOpAvgTime(node: Node|null|undefined) {
    if (node?.metrics?.avgTimePs) {
      return ` (${utils.formatDurationPs(node.metrics.avgTimePs)})`;
    }
    return '';
  }

  // Add avgTime info to the node svg
  updateGraphvizNodeText(element: Element) {
    const opName = this.getGraphvizNodeOpName(element);
    const node = this.getOpNodeInGraphviz(opName);
    if (!node) return;
    const svgEls = element.getElementsByTagName('text') || [];
    svgEls[0].textContent =
        `${svgEls[0].textContent} ${this.getOpAvgTime(node)}`;
  }

  // Add avgTime info to the cluster svg
  updateGraphvizClusterText(element: Element) {
    const opName = this.getGraphvizClusterOpName(element);
    const node = this.getOpNodeInGraphviz(opName);
    if (!node) return;
    const svgEls = element.getElementsByTagName('text') || [];
    svgEls[1].textContent =
        `${svgEls[1].textContent} ${this.getOpAvgTime(node)}`;
  }

  // Add runtime data (eg. AvgTime) to the graph node to help with perf
  // debugging.
  injectRuntimeData() {
    if (!this.opProfile || this.runtimeDataInjected ||
        !this.graphIframeLoaded()) {
      return;
    }
    const doc: Document|null = this.getGraphIframeDocument();
    if (!doc) return;
    const nodeElements = Array.from(doc.getElementsByClassName('node'));
    for (const e of nodeElements) {
      this.updateGraphvizNodeText(e);
    }
    const clusterElements = Array.from(doc.getElementsByClassName('cluster'));
    for (const e of clusterElements) {
      this.updateGraphvizClusterText(e);
    }
    this.runtimeDataInjected = true;
  }

  // Function called whenever user click the search graph button
  // Input params are passed from the graph config component.
  onSearchGraph(params?: Partial<GraphConfigInput>) {
    // update local variables with the new params
    if (params) {
      this.updateParams(params);
    }

    // rerouter instead of calling updateView directly to populate the url and
    // trigger re-parsing of the query params accordingly.
    this.router.navigate([], {
      relativeTo: this.route,
      queryParams: this.getGraphSearchParams(),
    });
  }

  // Event handler for module selection change in graph config form,
  // so we can handle the hlo text loading correctly.
  onModuleSelectionChange(moduleName: string) {
    this.selectedModule = moduleName;
    const regex = /\((.*?)\)/;
    const programIdMatch = this.selectedModule.match(regex);
    this.programId = programIdMatch ? programIdMatch[1] : '';
  }

  // Get a GraphConfigInput object for usage in the angular components.
  getParams(): GraphConfigInput {
    return {
      selectedModule: this.selectedModule,
      opName: this.opName,
      graphWidth: this.graphWidth,
      showMetadata: this.showMetadata,
      mergeFusion: this.mergeFusion,
      programId: this.programId,
      symbolId: this.symbolId,
      symbolType: this.symbolType,
      graphType: this.graphType,
    };
  }

  updateParams(param: Partial<GraphConfigInput>) {
    Object.entries(param).forEach(([key, value]) => {
      if (GRAPH_CONFIG_KEYS.includes(key)) {
        Object.assign(this, {[key]: value});
      }
    });
  }

  validToPlot() {
    // Validate opName
    if (
        // Parameter and ROOT node is not identified as an op in the HLO Graph
        this.opName.toLowerCase().includes('parameter') ||
        this.opName.toLowerCase().includes('root')) {
      this.openSnackBar('Invalid Op Name.');
      return false;
    }
    return this.opName && (this.selectedModule || this.programId);
  }

  onPlot() {
    if (!this.validToPlot()) return;
    // Always reset before new rendering
    this.resetPage();
    // - For graphvizHtml: clear the iframe so the `graphIframeLoaded`
    // detection is accurate
    // - If `show_me_graph` is true, render ModelExplorer Graph instead
    if (this.showMeGraph) {
    } else {
      this.renderGraphvizHtml();
    }
  }

  renderGraphvizHtml() {
    this.loadingGraph = true;
    const searchParams = new Map<string, string>();
    for (const [key, value] of Object.entries(this.getGraphSearchParams())) {
      searchParams.set(key, value);
    }
    this.tryRenderGraphvizHtml(searchParams);
  }

  // Get the query params to construct payload of the fetch request.
  getGraphSearchParams(): GraphViewerQueryParams {
    // Update the query parameters in url after form updates
    const queryParams: GraphViewerQueryParams = {
      'node_name': this.opName,
      'module_name': this.selectedModule,
      'graph_width': this.graphWidth,
      'show_metadata': this.showMetadata,
      'merge_fusion': this.mergeFusion,
      'graph_type': this.graphType,
    };
    if (this.programId !== '') {
      queryParams.program_id = this.programId;
    }
    if (this.symbolId !== '') {
      queryParams.symbol_id = this.symbolId;
    }
    if (this.symbolType !== '') {
      queryParams.symbol_type = this.symbolType;
    }
    if (this.showMeGraph) {
      queryParams.show_me_graph = true;
    }
    return queryParams;
  }

  tryRenderGraphvizHtml(searchParams: Map<string, string>) {
    const iframe = document.getElementById('graph-html') as HTMLIFrameElement;
    setTimeout(() => {
      if (!iframe) {
        this.tryRenderGraphvizHtml(searchParams);
      }
    }, 200);
    this.graphvizUri =
        this.dataService.getGraphVizUri(this.sessionId, searchParams);
    if (iframe?.contentWindow?.location) {
      locationReplace(
          iframe.contentWindow?.location,
          this.graphvizUri!,
      );
    }

    this.onCheckGraphHtmlLoaded();
  }

  getGraphIframeDocument() {
    return this.graphRef?.nativeElement?.contentDocument;
  }

  graphIframeLoaded() {
    const doc = this.getGraphIframeDocument();
    if (!doc) return false;
    // This is the feature we observed from html/svg generated by
    // third_party/tensorflow/compiler/xla/service/hlo_graph_dumper.cc to
    // determine if the graph has been loaded completedly.
    // We need add a test to detect the breaking change ahread.
    const loadingIdentifierNode = (doc.getElementsByTagName('head') || [])[0];
    return loadingIdentifierNode && loadingIdentifierNode.childElementCount > 0;
  }

  // Append diagnostic message after data loaded for each sections
  onCompleteLoad(diagnostics?: Diagnostics) {
    this.diagnostics = {
      info: [
        ...(diagnostics?.info || []),
      ],
      errors: [
        ...(diagnostics?.errors || []),
      ],
      warnings: [
        ...(diagnostics?.warnings || []),
      ],
    };
  }

  clearGraphIframeHtml() {
    const doc = this.getGraphIframeDocument();
    if (!doc) return;
    doc.firstElementChild?.remove();
  }

  onCheckGraphHtmlLoaded() {
    if (!this.graphIframeLoaded()) {
      setTimeout(() => {
        this.onCheckGraphHtmlLoaded();
      }, 1000);
      return;
    } else {
      this.loadingGraph = false;
      const htmlSize =
          (document.getElementById('graph-html') as HTMLIFrameElement)
              .contentDocument!.documentElement.innerHTML.length;
      if (htmlSize > GRAPH_HTML_THRESHOLD) {
        this.onCompleteLoad({
          warnings: [
            'Your graph is large. If you can\'t see the graph, please lower the width and retry.'
          ]
        } as Diagnostics);
      }
      this.installEventListeners();
      this.injectRuntimeData();
    }
  }

  // Resetting url, iframe, diagnostic messages per graph search
  resetPage() {
    // Clear iframe html so the rule to detect `graphIframeLoaded` can satisfy
    this.clearGraphIframeHtml();
    this.diagnostics = {...DIAGNOSTICS_DEFAULT};
  }

  ngOnDestroy() {
    // Unsubscribes all pending subscriptions.
    this.destroyed.next();
    this.destroyed.complete();
  }
}
