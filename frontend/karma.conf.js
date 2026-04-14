/**
 * Karma configuration.
 * @param {!Object} config
 */
module.exports = function(config) {
  config.proxies = {
    '/worker_bin.js': '/base/learning/brain/mobile/lite/tooling/model_graph_visualizer/worker_bin.js',
    '/base/third_party/xprof/frontend/materialicons.woff2': '/base/third_party/xprof/plugin/xprof/static/materialicons.woff2',
  };
  config.files.push({
    pattern: 'https://www.gstatic.com/charts/loader.js',
    watched: false,
    included: true,
    served: false,
  });
};
