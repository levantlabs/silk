* can ngspice report each instance's drawn model parameters?
.lib /synthetic/pdk/sky130.lib.spice tt_mm
.options seed=7
i1 0 g1 20u
i2 0 g2 20u
xm1 g1 g1 0 0 sky130_fd_pr__nfet_01v8 w=5 l=2
xm2 g2 g2 0 0 sky130_fd_pr__nfet_01v8 w=5 l=2
.control
op
print v(g1)-v(g2)
showmod m.xm1.msky130_fd_pr__nfet_01v8 : vth0 toxe voff
showmod m.xm2.msky130_fd_pr__nfet_01v8 : vth0 toxe voff
.endc
.end
