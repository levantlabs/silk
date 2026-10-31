* TASK-330.3 probe: post-run '@' save + explicit op, on an .ac bench.

.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.subckt dut d g s b
XMmmn d g s b sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15
.ends dut

xdut vdn vgn 0 0 dut

vgn vgn 0 dc 1.0 ac 1.0
vdn vdn 0 dc 1.0

.ac dec 10 1 1e9

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
write ac_op_after.raw ac1.all op1.all
display
setplot
exit
.endc
.end
