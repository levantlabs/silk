"""gm/Id lookup curves for sky130 NMOS and PMOS devices.

A gate-voltage DC sweep supplies gm, id, gds, cgg, vgs and vth probes.
Signal arithmetic forms gm/Id, Id/W, gm/gds, the estimate ft = gm/(2*pi*cgg),
and Vov = vgs - vth. Sign-corrected probes use the same Vov convention for
NMOS and PMOS; Vov can be negative below threshold.

Curves are approximately width-independent. The example checks width
sensitivity and refines the sizing read-off at the estimated width to
account for model width dependence. An independent operating-point run
checks gm/Id at one bias and the gm achieved by the selected width.
Nonmonotonic axes remain parametric traces and are reported in notes.

Run ``MPLBACKEND=Agg .venv/bin/python -m silk.examples.gm_id_lookup``.
It prints sizing read-offs and checks, then saves a self-contained HTML
page with four plots per device under example_output_dir(). Needs ngspice
and sky130 models.
"""
from __future__ import annotations

import base64
import io
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from silk import Module, load_pdk, sources
from silk.circuit.circuit_elements import Port
from silk.examples import example_output_dir
from silk.simulation import DC, Op

VDD_V = 1.8
VDS_V = 0.9
W_UM = 2.0
L_LIST_UM = (0.15, 0.5)
W_CHECK_UM = 8.0
DESIGN_REFINEMENTS = 1
VG_STEP_V = 0.01
DEVICES = {'NMOS': ('nmos1p8', 0.0), 'PMOS': ('pmos1p8', VDD_V)}

L_DESIGN_UM = 0.5
GM_ID_DESIGN = 15.0
GM_TARGET_S = 100e-6
W_CHECK_GM_ID = (8.0, 12.0, 16.0)

HAND_CHECK_VGS_V = 0.6
HAND_CHECK_RTOL = 1e-6


def _bench(dev, l_um, w_um, vg, analysis):
    """One transistor, rail = source = body, drain held at |Vds| = VDS_V."""
    cell, rail = DEVICES[dev]
    pdk = load_pdk('sky130')
    mod = Module('sky130', example_output_dir('gm_id_lookup'), 'gmid.cir', pdk_details=pdk,
                 module_name='gmid')
    mod.define_module_ports({
        'D': Port.analog_output(), 'G': Port.analog_input(),
        'RAIL': Port.supply() if rail else Port.ground(),
    })
    with mod:
        getattr(pdk, cell)('M1', d='D', g='G', s='RAIL', b='RAIL',
                           w=w_um, l=l_um)
    tb = mod.create_test('gmid')
    tb.add_module_under_test({'D': 'd', 'G': 'g', 'RAIL': 'rail'})
    drain = rail + VDS_V if rail == 0 else rail - VDS_V
    tb.add_source(sources.VoltageSource('rail', positive='rail', negative='0', dc=rail))
    tb.add_source(sources.VoltageSource('vd', positive='d', negative='0', dc=drain))
    tb.add_source(sources.VoltageSource('vg', positive='g', negative='0', dc=vg))
    tb.add_sim_type(analysis)
    return mod, tb


def sweep(dev, l_um, w_um):
    """DC sweep of the gate; returns the probed Signals (bound after the run)."""
    mod, tb = _bench(dev, l_um, w_um, 0.0,
                     DC(source='Vvg', start=0.0, stop=VDD_V, step=VG_STEP_V))
    p = {k: tb.probe(f'M1:{k}') for k in ('gm', 'id', 'gds', 'cgg', 'vgs', 'vth')}
    p['vg'] = tb.probe('g')
    mod.run_test('gmid', corner='tt')
    return {k: sig.snapshot() for k, sig in p.items()}


def op_point(dev, l_um, w_um, vg):
    """The op_table() row of one transistor at a single bias."""
    mod, tb = _bench(dev, l_um, w_um, vg, Op())
    mod.save_device_params = True
    report = mod.run_test('gmid', corner='tt')
    return report.op_table()[0]


def lookup_views(p, w_um):
    """The four lookup views of one sweep, as Signals, plus the raw inputs.

    A view whose x (gm/Id or Id/W) is not strictly monotonic over the whole
    sweep cannot be a function graph; it is kept as a parametric trace and
    recorded in ``notes``.
    """
    gm_id = p['gm'] / p['id']
    gm_gds = p['gm'] / p['gds']
    ft = p['gm'] / (p['cgg'] * (2 * math.pi))
    id_w = p['id'] / w_um
    vov = p['vgs'] - p['vth']
    notes = []

    def versus(y, x, label, axis):
        try:
            return y.versus(x, name=label)
        except ValueError as exc:
            d = np.diff(x.y)
            opposed = np.abs(d[np.sign(d) != np.sign(d.sum())]).sum()
            msg = (f'{exc} Opposing steps total '
                   f'{opposed / np.ptp(x.y):.1%} of the {axis} range: {axis} '
                   f'is not a single-valued key for this device.')
            if msg not in notes:
                notes.append(msg)
            return y.versus(x, name=label, allow_nonmonotonic=True)

    return {
        'gm_id': gm_id, 'id_w': id_w, 'gm_gds': gm_gds, 'ft': ft, 'vov': vov,
        'a': versus(gm_id, id_w, 'gm/Id vs Id/W', 'Id/W'),
        'b': versus(gm_gds, gm_id, 'gm/gds vs gm/Id', 'gm/Id'),
        'c': versus(ft, gm_id, 'ft vs gm/Id', 'gm/Id'),
        'd': versus(vov, gm_id, 'Vov vs gm/Id', 'gm/Id'),
        'notes': notes,
    }


def _strong_side(gm_id_sig, vg_sig, dev):
    """Gate-voltage window over which gm/Id is strictly monotonic and reaches
    the strong-inversion end, so the design read-off interpolates safely.
    """
    g, vg = gm_id_sig.y, vg_sig.y
    # Traverse from strong toward weak inversion.
    seq = np.arange(len(g))[::-1] if dev == 'NMOS' else np.arange(len(g))
    k = 1
    while k < len(seq) and g[seq[k]] > g[seq[k - 1]]:
        k += 1
    lo, hi = sorted((vg[seq[0]], vg[seq[k - 1]]))
    return lo, hi


def _at_gm_id(p, views, dev, gm_id_values):
    """Id/W, gm/gds and the gate voltage read off at the given gm/Id values,
    over the strong-inversion window where gm/Id is strictly monotonic.
    """
    lo, hi = _strong_side(views['gm_id'], p['vg'], dev)
    gm_id = views['gm_id'].window(lo, hi)
    at = list(gm_id_values)

    def read(sig):
        return sig.window(lo, hi).versus(gm_id).interpolate(at).y

    return read(views['id_w']), read(views['gm_gds']), read(p['vg'])


def readoff(p, views, dev):
    """Design read-off at gm/Id = GM_ID_DESIGN and gm = GM_TARGET_S."""
    id_w, gain, vg = _at_gm_id(p, views, dev, [GM_ID_DESIGN])
    id_need = GM_TARGET_S / GM_ID_DESIGN
    return {'id_w': float(id_w[0]), 'gain': float(gain[0]), 'vg': float(vg[0]),
            'id': id_need, 'w_um': id_need / float(id_w[0])}


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def _figure(dev, per_l):
    fig, ax = plt.subplots(2, 2, figsize=(10, 7.5))

    specs = (
        ('a', ax[0, 0], 'Id/W (uA/um)', 1e6, 'gm/Id (S/A)', 1.0, True, False),
        ('b', ax[0, 1], 'gm/Id (S/A)', 1.0, 'gm/gds', 1.0, False, False),
        ('c', ax[1, 0], 'gm/Id (S/A)', 1.0, 'ft = gm/(2 pi cgg) (GHz)', 1e-9, False, True),
        ('d', ax[1, 1], 'gm/Id (S/A)', 1.0, 'Vov = vgs - vth (V)', 1.0, False, False),
    )
    for key, a, xl, xs, yl, ys, logx, logy in specs:
        for l_um, views in per_l.items():
            views[key].plot(ax=a, logx=logx, label=f'L = {l_um} um')

        # Scale the drawn lines while keeping the read-off Signals in SI units.
        for line in a.lines:
            line.set_data(np.asarray(line.get_xdata()) * xs,
                          np.asarray(line.get_ydata()) * ys)
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        if logx:
            a.set_xscale('log', nonpositive='mask')
        if logy:
            a.set_yscale('log', nonpositive='mask')
        a.relim()
        a.autoscale_view()
        a.grid(True, alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(f'sky130 {dev} lookup curves (W = {W_UM:g} um, |Vds| = {VDS_V} V)')
    fig.tight_layout()
    return _png(fig)


def _html(images, rows, checks, notes):
    figs = ''.join(
        f'<h2>{d}</h2><img src="data:image/png;base64,{b}" alt="{d} lookup views">'
        for d, b in images.items())
    trs = ''.join(
        f'<tr><td>{r["dev"]}</td><td>{r["l"]}</td><td>{r["id_w"] * 1e6:.2f}</td>'
        f'<td>{r["gain"]:.1f}</td><td>{r["vg"]:.3f}</td><td>{r["w_um"]:.1f}</td>'
        f'<td>{r["op_gm"] * 1e6:.1f}</td></tr>' for r in rows)
    chk = ''.join(f'<li>{c}</li>' for c in checks)
    nts = ''.join(f'<li>{n}</li>' for n in notes) or '<li>none</li>'
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>gm/Id lookup curves - sky130</title>
<style>
 body{{font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:960px;
      margin:2rem auto;padding:0 1rem;color:#1a1a1a;background:#fafafa}}
 img{{width:100%;border:1px solid #ddd;border-radius:6px;background:#fff}}
 table{{border-collapse:collapse;margin:1rem 0;font-size:14px}}
 td,th{{border:1px solid #ddd;padding:.35rem .7rem;text-align:left}}
 th{{background:#f0f0f4}}
</style></head><body>
<h1>gm/Id lookup curves - sky130 NMOS and PMOS</h1>
<p>From device-parameter probes (<code>tb.probe('M1:gm')</code>) over a DC sweep
of the gate; W = {W_UM:g} um, |Vds| = {VDS_V} V. PMOS id and gm use
sign-corrected values; Vov can be negative below threshold.
<code>ft</code> is the estimate gm/(2 pi cgg).</p>
{figs}
<h2>Design read-off: gm/Id = {GM_ID_DESIGN:g} S/A, gm = {GM_TARGET_S * 1e6:g} uS</h2>
<table><tr><th>device</th><th>L (um)</th><th>Id/W (uA/um)</th><th>gm/gds</th>
<th>|Vgs| (V)</th><th>W (um)</th><th>gm of that W, Op run (uS)</th></tr>{trs}</table>
<h2>Checks</h2><ul>{chk}</ul>
<h2>Views kept as parametric traces (x not strictly monotonic)</h2><ul>{nts}</ul>
</body></html>"""


def main():
    images, rows, checks, notes = {}, [], [], []
    for dev, (_cell, rail) in DEVICES.items():
        per_l = {}
        for l_um in L_LIST_UM:
            p = sweep(dev, l_um, W_UM)
            v = lookup_views(p, W_UM)
            assert p['id'].y.min() >= 0 and p['gm'].y.min() >= 0, \
                f'{dev}: id/gm must be positive (sign-corrected)'
            per_l[l_um] = v
            notes += [f'{dev}, L = {l_um} um: {n}' for n in v['notes']]
            if l_um == L_DESIGN_UM:
                design = (p, v)
                print(f'\n{dev} L={l_um}: gm/Id {v["gm_id"].y.max():.1f} '
                      f'(weak) .. {v["gm_id"].y.min():.2f} (strong) S/A')
        images[dev] = _figure(dev, per_l)

        p, v = design
        r = readoff(p, v, dev)
        for _ in range(DESIGN_REFINEMENTS):

            # Re-sweep at the estimated width because the models depend on width.
            w_prev = r['w_um']
            pr = sweep(dev, L_DESIGN_UM, w_prev)
            r = readoff(pr, lookup_views(pr, w_prev), dev)
            print(f'  refined at W={w_prev:.1f} um -> W={r["w_um"]:.1f} um')
        op = op_point(dev, L_DESIGN_UM, r['w_um'], r['vg'])

        vgs = r['vg'] if dev == 'NMOS' else VDD_V - r['vg']
        print(f'  read-off gm/Id={GM_ID_DESIGN:g}: |Vgs|={vgs:.3f} V  '
              f'Id/W={r["id_w"] * 1e6:.2f} uA/um  gm/gds={r["gain"]:.1f}  '
              f'-> W={r["w_um"]:.1f} um for gm={GM_TARGET_S * 1e6:g} uS')
        print(f'  Op run of that W: gm={op.gm * 1e6:.1f} uS  gm/Id={op.gm / op.id:.2f}')
        checks.append(f'{dev} loop closure: Op run at W = {r["w_um"]:.1f} um reports '
                      f'gm = {op.gm * 1e6:.1f} uS (asked {GM_TARGET_S * 1e6:g} uS), after '
                      f'{DESIGN_REFINEMENTS} re-sweep at the W read off (W-binned models).')
        rows.append({'dev': dev, 'l': L_DESIGN_UM, 'id_w': r['id_w'],
                     'gain': r['gain'], 'vg': vgs, 'w_um': r['w_um'],
                     'op_gm': op.gm})

        vg_hc = HAND_CHECK_VGS_V if dev == 'NMOS' else VDD_V - HAND_CHECK_VGS_V
        i = int(np.argmin(np.abs(p['vg'].y - vg_hc)))
        assert abs(p['vg'].y[i] - vg_hc) < 1e-9
        hc = op_point(dev, L_DESIGN_UM, W_UM, vg_hc)
        sweep_gmid = float(v['gm_id'].y[i])
        op_gmid = hc.gm / hc.id
        err = abs(sweep_gmid - op_gmid) / op_gmid
        print(f'  hand check |Vgs|={HAND_CHECK_VGS_V} V: sweep gm/Id={sweep_gmid:.6f}  '
              f'op_table gm={hc.gm:.6e} id={hc.id:.6e} -> gm/Id={op_gmid:.6f}  '
              f'rel.err={err:.2e}')
        assert err < HAND_CHECK_RTOL, f'{dev} hand check failed: {err:.3e}'
        checks.append(f'{dev} hand check at |Vgs| = {HAND_CHECK_VGS_V} V: sweep gm/Id = '
                      f'{sweep_gmid:.6f}; op_table() gm/id = {op_gmid:.6f} '
                      f'(rel. diff {err:.1e}).')

        pw = sweep(dev, L_DESIGN_UM, W_CHECK_UM)
        vw = lookup_views(pw, W_CHECK_UM)
        ref = _at_gm_id(p, v, dev, W_CHECK_GM_ID)[0]
        alt = _at_gm_id(pw, vw, dev, W_CHECK_GM_ID)[0]
        d_idw = float(np.max(np.abs(alt / ref - 1)))
        print(f'  W independence (W={W_CHECK_UM:g} vs {W_UM:g} um): Id/W at gm/Id = '
              f'{W_CHECK_GM_ID} differs by {d_idw:.2%} at most')
        checks.append(f'{dev} W independence: Id/W read at gm/Id = {W_CHECK_GM_ID} S/A from '
                      f'W = {W_CHECK_UM:g} um vs {W_UM:g} um differs by {d_idw:.2%} at most '
                      f'(the PDK models are binned by W, so it is close, not exact).')

    out = example_output_dir() + '/gm_id_lookup.html'
    with open(out, 'w') as f:
        f.write(_html(images, rows, checks, notes))
    print(f'\n  saved {out}')
    for n in notes:
        print(f'  note: {n}')
    return out


if __name__ == '__main__':
    main()
