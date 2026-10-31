* Isolates whether cgso/cgdo/cgbo's 42-vs-47 divergence (found in
* op_all17_probe.sp, a 2-level dut-wrapped hierarchy) is a hierarchy-depth
* effect: same bias/geometry as tests/test_op_point_params.py's
* _single_nmos_tb (FLAT top-level XM1, no dut wrapper), pre-run save
* (probe style).
.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

XM1 vd vg 0 0 sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15

vg vg 0 dc 1.0
vd vd 0 dc 1.0

.control
save all
save @m.xm1.msky130_fd_pr__nfet_01v8[cgso]
save @m.xm1.msky130_fd_pr__nfet_01v8[cgdo]
save @m.xm1.msky130_fd_pr__nfet_01v8[cgbo]
save @m.xm1.msky130_fd_pr__nfet_01v8[gm]
op
run
print @m.xm1.msky130_fd_pr__nfet_01v8[cgso]
print @m.xm1.msky130_fd_pr__nfet_01v8[cgdo]
print @m.xm1.msky130_fd_pr__nfet_01v8[cgbo]
print @m.xm1.msky130_fd_pr__nfet_01v8[gm]
exit
.endc
.end
