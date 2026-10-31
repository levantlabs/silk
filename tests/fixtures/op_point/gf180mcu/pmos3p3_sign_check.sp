* TASK-330.4 ruling (D): verify gf180mcu pfet_03v3 '@' sign convention at
* the SAME bias 330.1 used for sky130 pfet_01v8 (Vg=0.3, Vd=0, Vs=Vb=1.8),
* to confirm ngspice reports vgs/vds/id as positive source-referenced
* magnitudes here too (330.1's sky130 finding), or with raw physical signs.

.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" res_typical
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" moscap_typical
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" mimcap_typical
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" cap_mim
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" diode_typical
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" bjt_typical
.include "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/design.ngspice"

XMM1 vd vg vs vs pfet_03v3 m=1 w=10u l=0.28u

Vvs vs 0 DC 1.8
Vvd vd 0 DC 0
Vvg vg 0 DC 0.3

.op

.control
save all
op
print @m.xmm1.m0[vgs]
print @m.xmm1.m0[vds]
print @m.xmm1.m0[id]
print @m.xmm1.m0[vth]
print @m.xmm1.m0[vdsat]
write pmos3p3_sign_check.raw op1.all
exit
.endc
.end
