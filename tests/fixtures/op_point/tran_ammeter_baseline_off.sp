* TASK-330.3 probe: same ammeter bench, switch OFF -- baseline for diffing tran1.

.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.subckt amm_sky130_fd_pr__nfet_01v8 d g s b m=1 w=1 l=0.15
Vamm_d d d_i 0
Vamm_g g g_i 0
Vamm_s s s_i 0
Vamm_b b b_i 0
XMdev d_i g_i s_i b_i sky130_fd_pr__nfet_01v8 m={m} w={w} l={l}
.ends

.subckt dut d g s b
XMmm1 d g s b amm_sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15
.ends

xdut vd vg vs vs dut

Vvd vd 0 DC 1
Vvg vg 0 DC 1
Vvs vs 0 DC 0

.options savecurrents
.tran 1n 10n

.control
save all
run
set wr_singlescale
set wr_vecnames
option numdgt=3
write tran_ammeter_baseline_off.raw
exit
.endc
.end
