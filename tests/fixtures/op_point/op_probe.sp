* TASK-330.1 verification deck: operating-point device-parameter '@' vectors.
* Wrapper subckt around sky130_fd_pr__nfet_01v8 and sky130_fd_pr__pfet_01v8,
* mirroring the hierarchy silk itself emits (a DUT subckt instanced as
* 'xdut', placing the PDK device as an 'XM<name>' X-element), so the '@'
* vector path this deck exercises is the exact one op_table's real decks
* will use: @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm].
*
* NFET: Vg=1.0 Vd=1.0 Vs=0 Vb=0 (Vgs=1.0V, comfortably above Vth, Vds=1.0V
* -> saturation).
* PFET: Vs=1.8 Vb=1.8 Vg=0.3 Vd=0 (absolute node voltages exactly as the
* task asks: "at Vg=0.3, Vd=0" -> Vsg=1.5V, Vsd=1.8V -> saturation).

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
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gds]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gmbs]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vth]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vdsat]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[id]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vgs]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vds]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vbs]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgg]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgs]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgd]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgso]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[cgdo]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[gm]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[gds]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vth]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vdsat]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[id]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vgs]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vds]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vbs]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgg]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgs]
save @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[cgd]
op
run
display
set altshow
show m.xdut.xmmmn.msky130_fd_pr__nfet_01v8 : gm gds gmbs vth vdsat id vgs vds vbs cgg cgs cgd
show m.xdut.xmmmp.msky130_fd_pr__pfet_01v8 : gm gds gmbs vth vdsat id vgs vds vbs cgg cgs cgd
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vgs]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vds]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vbs]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[id]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vgs]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vds]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[vbs]
print @m.xdut.xmmmp.msky130_fd_pr__pfet_01v8[id]
write op_probe.raw
exit
.endc
.end
