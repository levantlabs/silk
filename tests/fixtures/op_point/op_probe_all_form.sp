* TASK-330.1 check (6): does 'save @m.<path>[all]' work as a short form for
* every BSIM4 op-var on that device, instead of one 'save' line per param?

.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.subckt dut dn gn sn bn
XMmmn dn gn sn bn sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15
.ends dut

xdut vdn vgn 0 0 dut

vgn vgn 0 dc 1.0
vdn vdn 0 dc 1.0

.control
save all
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[all]
op
run
display
exit
.endc
.end
