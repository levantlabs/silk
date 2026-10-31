* TASK-330.3 probe: post-run '@' save + explicit op, on a DC() sweep bench
* (multi-point, not the single-point op-like case).

.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.subckt dut d g s b
XMmmn d g s b sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15
.ends dut

xdut vdn vgn 0 0 dut

vgn vgn 0 dc 1.0
vdn vdn 0 dc 1.0

.dc vgn 0 1.8 0.2

.control
save all
run
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[id]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vth]
op
set wr_singlescale
set wr_vecnames
option numdgt=3
write dc_op_after.raw dc1.all op1.all
display
setplot
exit
.endc
.end
