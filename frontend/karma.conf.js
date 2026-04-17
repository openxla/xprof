/**
 * Karma configuration.
 * @param {!Object} config
 */
module.exports = function(config) {
  config.proxies = {
    '/worker_bin.js': '/base/learning/brain/mobile/lite/tooling/model_graph_visualizer/worker_bin.js'
  };
  config.files.push({
    pattern: 'https://www.gstatic.com/charts/loader.js',
    watched: false,
    included: true,
    served: false,
  });
};
