/* Standalone compatibility: the mountable renderer owns no window globals. */
(function () {
  'use strict';
  var transport = {
    request: function (src, slot, onerror) {
      var script = document.createElement('script');
      script.onerror = onerror;
      script.src = src + '?_=' + Date.now();
      slot.replaceChildren(script);
    },
  };
  Object.defineProperties(transport, {
    data: { get: function () { return window.__silkViewData; },
      set: function (handler) { window.__silkViewData = handler; } },
    stamp: { get: function () { return window.__silkViewStamp; },
      set: function (handler) { window.__silkViewStamp = handler; } },
  });
  transport.data = function () {};
  transport.stamp = function () {};
  var queryCommands = [], queryError = null;
  try {
    var m = /(?:^|[?&])commands=([^&]*)/.exec(location.search);
    if (m) {
      queryCommands = JSON.parse(decodeURIComponent(m[1]));
      if (!Array.isArray(queryCommands)) throw new Error('commands must be an array');
    }
  } catch (error) { queryCommands = []; queryError = '?commands= is not valid JSON — ignored'; }
  var base = (location.pathname || '').split('/').pop() || 'view.html';
  var instance = createExplorer(document.body, {
    uPlot: window.uPlot, waveformCore: window.SILK_WAVEFORM_CORE,
    config: window.SILK_VIEW_CONFIG, scopes: window.SILK_VIEW_SCOPES,
    viewState: window.SILK_VIEW_STATE, commands: window.SILK_VIEW_COMMANDS,
    // TASK-319: forwards emit_view(group_signals_by_kind=)'s value, baked
    // into SILK_VIEW_CONFIG the same way poll_ms/inline already are, as the
    // explorer's own groupSignalsByKind option (the standalone page's half
    // of that single setting).
    groupSignalsByKind: (window.SILK_VIEW_CONFIG || {}).group_by_kind !== false,
    queryCommands: queryCommands, transport: transport,
    standaloneStateKey: function (payload) { return payload.run.state_key; },
    viewFilename: (base.replace(/\.[^./]*$/, '') || 'view') + '.silkview.json',
    onTitle: function (title) { document.title = title; },
    onPayload: function (payload) { window.SILK_VIEW_DATA = payload; },
  });
  document.body.querySelectorAll('[data-silk]').forEach(function (node) { node.id = node.dataset.silk; });
  window.silkView = { run: instance.run, commands: instance.commands, version: instance.version };
  function fatal(message) {
    var banner = document.getElementById('fatal-banner');
    banner.textContent = message; banner.hidden = false;
    document.getElementById('app').style.display = 'none';
  }
  if (!window.SILK_VIEW_DATA) {
    fatal('No waveform data was found (window.SILK_VIEW_DATA is missing). This page must be opened alongside its ' +
      ((window.SILK_VIEW_CONFIG || {}).data_src || 'data.js') + ' sidecar, or built with --inline.');
  } else {
    try {
      instance.load(window.SILK_VIEW_DATA);
      if (queryError) {
        var toast = document.createElement('div'); toast.className = 'toast'; toast.textContent = queryError;
        document.getElementById('toast-container').appendChild(toast);
        setTimeout(function () { toast.remove(); }, 4000);
      }
    } catch (error) { fatal(error.message); }
  }
})();
