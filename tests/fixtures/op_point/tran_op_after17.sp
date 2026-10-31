* TASK-330.3 probe: post-run '@' save (all 17 params) + explicit op, tran bench.

.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.subckt dut d g s b
XMmmn d g s b sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15
.ends dut

xdut vdn vgn 0 0 dut

vgn vgn 0 dc 1.0
vdn vdn 0 dc 1.0

.tran 1u 10u

.control
save all
run
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
op
set wr_singlescale
set wr_vecnames
option numdgt=3
write tran_op_after17.raw tran1.all op1.all
display
setplot
exit
.endc
.end
