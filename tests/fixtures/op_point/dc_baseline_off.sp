* TASK-330.3 probe: same DC() bench, switch OFF -- baseline for diffing dc1.

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
set wr_singlescale
set wr_vecnames
option numdgt=3
write dc_baseline_off.raw
exit
.endc
.end
