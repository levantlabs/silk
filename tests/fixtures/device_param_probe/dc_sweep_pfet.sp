* TASK-330.8 AC#1 spike: DC-sweep device-parameter probing, PMOS.
* Absolute biasing as TASK-330.1 (Vs=Vb=1.8V, Vd=0V), Vg swept 1.8V->0V so
* Vsg goes 0V->1.8V. All 17 DEVICE_OP_PARAMS saved BEFORE '.dc' runs.

.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.subckt dut dp gp sp bp
XMmmp dp gp sp bp sky130_fd_pr__pfet_01v8 m=1 w=1 l=0.15
.ends dut

xdut vdp vgp vsp vsp dut

vsp vsp 0 dc 1.8
vgp vgp 0 dc 0.9
vdp vdp 0 dc 0

.dc vgp 1.8 0 -0.05

.control
save all
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
run
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
write dc_sweep_pfet.raw
exit
.endc
.end
