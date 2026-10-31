* TASK-330.8 AC#1 spike: DC-sweep device-parameter probing.
* NFET, same wrapper hierarchy as tests/fixtures/op_point/op_probe.sp
* (dut subckt -> xdut -> XMmmn -> sky130_fd_pr__nfet_01v8), Vds fixed at
* 1.0V, Vgs swept 0V->1.8V so id/gm/vth/... vary across the sweep the way
* a probe used against a DC-sweep analysis would need them to.
* All 17 DEVICE_OP_PARAMS saved BEFORE '.dc' runs (per-probe capture, not
* the post-run op-plot capture save_device_params uses).

.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.subckt dut dn gn sn bn
XMmmn dn gn sn bn sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15
.ends dut

xdut vdn vgn 0 0 dut

vgn vgn 0 dc 0.9
vdn vdn 0 dc 1.0

.dc vgn 0 1.8 0.05

.control
save all
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[id]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vgs]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vds]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vbs]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vth]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vdsat]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gds]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gmbs]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgg]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgs]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgd]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgso]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgdo]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgbo]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[capbd]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[capbs]
run
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[id]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vgs]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vds]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vbs]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vth]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vdsat]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gds]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gmbs]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgg]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgs]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgd]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgso]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgdo]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgbo]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[capbd]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[capbs]
set wr_singlescale
set wr_vecnames
option numdgt=6
write dc_sweep_nfet.raw
exit
.endc
.end
