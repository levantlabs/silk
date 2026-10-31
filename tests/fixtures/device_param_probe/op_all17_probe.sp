* TASK-330.8 AC#1 spike: op-point probing of the FULL 17-param
* DEVICE_OP_PARAMS list (330.1's op_probe.sp only covered 12-14; this
* fixture adds cgbo/capbd/capbs and PFET gmbs/cgso/cgdo/cgbo/capbd/capbs
* so every column's op-point behaviour is on record, esp. the overlap
* caps' "reads 0 outside .tran" claim).
* Same bias as op_probe.sp: NFET Vg=1.0 Vd=1.0 Vs=Vb=0; PFET Vs=Vb=1.8
* Vg=0.3 Vd=0 (absolute).

.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.subckt dut dn gn sn bn dp gp sp bp
XMmmn dn gn sn bn sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15
XMmmp dp gp sp bp sky130_fd_pr__pfet_01v8 m=1 w=1 l=0.15
.ends dut

xdut vdn vgn 0 0 vdp vgp vsp vsp dut

vgn vgn 0 dc 1.0
vdn vdn 0 dc 1.0
vsp vsp 0 dc 1.8
vgp vgp 0 dc 0.3
vdp vdp 0 dc 0

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
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[id]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vgs]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vds]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vbs]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vth]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vdsat]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[gm]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[gds]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[gmbs]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgg]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgs]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgd]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgso]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgdo]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgbo]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[capbd]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[capbs]
op
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
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[id]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vgs]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vds]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vbs]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vth]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vdsat]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[gm]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[gds]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[gmbs]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgg]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgs]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgd]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgso]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgdo]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgbo]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[capbd]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[capbs]
set wr_singlescale
set wr_vecnames
option numdgt=6
write op_all17_probe.raw
exit
.endc
.end
