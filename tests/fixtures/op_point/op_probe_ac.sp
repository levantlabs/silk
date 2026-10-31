* TASK-330.1 check (5): '@' device-param vectors are empty after a pure .ac
* analysis, and an explicit 'op' before 'run' restores a valid op1 plot
* alongside ac1. Same DUT/bias as op_probe.sp (NFET only, to keep it short).

.lib "$PDK_ROOT/sky130A/libs.tech/ngspice/sky130.lib.spice" tt

.subckt dut dn gn sn bn
XMmmn dn gn sn bn sky130_fd_pr__nfet_01v8 m=1 w=1 l=0.15
.ends dut

xdut vdn vgn 0 0 dut

vgn vgn 0 dc 1.0 ac 1.0
vdn vdn 0 dc 1.0

.ac dec 10 1 1e9

.control
save all
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm]
save @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vth]
* --- AC-only run first: '@' vectors should be empty in ac1 ---
ac dec 10 1 1e9
echo "--- after ac dec (no explicit op) ---"
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vth]
* --- now run an explicit op first, then ac again ---
op
echo "--- after explicit op (before ac) ---"
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm]
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[vth]
ac dec 10 1 1e9
echo "--- after the second ac (op1 should still be the last op result, ac1 the ac sweep) ---"
print @m.xdut.xmmmn.msky130_fd_pr__nfet_01v8[gm]
display
setplot
write op_probe_ac.raw
exit
.endc
.end
