* TASK-330.3 probe Q5, THE FIX (reviewer stop-point ruling 2, condition 1):
* Op() declared alongside Transient(), but with NO netlist-level '.op' card
* at all -- confirming generate_analysis_commands' suppression rule
* (save_device_params skips an Op() instance's own '.op' card unconditionally).
* The FIRST version of this fixture (superseded) still had the '.op' card and
* demonstrated the bug it causes: a second, phantom 'op2' plot with NONE of
* the '@' params, while 'op1' stays the param-less plot the '.op' card itself
* produced during the first 'run' -- so 'write ... op1.all' silently captured
* the WRONG, empty-of-params plot. See git history for that version.
*
* This version: the '.op' card is gone. Only the interactive post-run
* sequence produces an op-point plot at all, so it is 'op1' by construction --
* one op plot, with every '@' param active, no ambiguity.

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
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vth]
op
set wr_singlescale
set wr_vecnames
option numdgt=3
write tran_plus_op_fixed.raw tran1.all op1.all
display
setplot
exit
.endc
.end
