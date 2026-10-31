* Same flat XM1 device, transient, checking whether cgso/cgdo/cgbo's tran
* value (both ngspice versions) matches ngspice47's op/dc/ac value from
* flat_hierarchy_check.sp (2.463135e-16 / 2.463135e-16 / 1.302418e-20).
.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

XM1 vd vg 0 0 sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15

vg vg 0 dc 1.0
vd vd 0 dc 1.0

.tran 1n 5n

.control
save all
save @m.xm1.msky130_fd_pr__nfet_01v8[cgso]
save @m.xm1.msky130_fd_pr__nfet_01v8[cgdo]
save @m.xm1.msky130_fd_pr__nfet_01v8[cgbo]
run
print @m.xm1.msky130_fd_pr__nfet_01v8[cgso]
print @m.xm1.msky130_fd_pr__nfet_01v8[cgdo]
print @m.xm1.msky130_fd_pr__nfet_01v8[cgbo]
exit
.endc
.end
