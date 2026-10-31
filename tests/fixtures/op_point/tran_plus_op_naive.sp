* TASK-330.3 probe Q5: Op() declared ALONGSIDE Transient() (both netlist-level
* cards), '@' saves BEFORE 'run' (the naive/rejected placement) -- does tran1
* still get contaminated even though Op() is explicitly declared too?

.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.subckt dut d g s b
XMmmn d g s b sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15
.ends dut

xdut vdn vgn 0 0 dut

vgn vgn 0 dc 1.0
vdn vdn 0 dc 1.0

.tran 1u 10u
.op

.control
save all
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[id]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vth]
run
set wr_singlescale
set wr_vecnames
option numdgt=3
write tran_plus_op_naive.raw tran1.all op1.all
display
setplot
exit
.endc
.end
