* TASK-330.3 gate-finding follow-up: measured op-param availability for
* gf180mcu's pmos3p3 model (see gf180mcu_nmos3p3_op_params.sp's header
* for the internal-instance-name correction this deck depends on: 'm0',
* not 'm'+model-name). 14/17 DEVICE_OP_PARAMS resolve cleanly;
* cgso/cgdo/cgbo are unavailable/zero-length on this model and excluded.

.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" typical
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" res_typical
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" moscap_typical
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" mimcap_typical
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" cap_mim
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" diode_typical
.lib "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/sm141064.ngspice" bjt_typical
.include "$PDK_ROOT/gf180mcuD/libs.tech/ngspice/design.ngspice"

XMM1 vd vg vdd vdd pfet_03v3 m=1 w=10u l=0.28u

Vvdd vdd 0 DC 3.3
Vvd vd 0 DC 1.8
Vvg vg 0 DC 1.8

.tran 1n 5n

.control
save all
run
save @m.xmm1.m0[id]
save @m.xmm1.m0[vgs]
save @m.xmm1.m0[vds]
save @m.xmm1.m0[vbs]
save @m.xmm1.m0[vth]
save @m.xmm1.m0[vdsat]
save @m.xmm1.m0[gm]
save @m.xmm1.m0[gds]
save @m.xmm1.m0[gmbs]
save @m.xmm1.m0[cgg]
save @m.xmm1.m0[cgs]
save @m.xmm1.m0[cgd]
save @m.xmm1.m0[capbd]
save @m.xmm1.m0[capbs]
op
write pmos3p3_op_params.raw tran1.all op1.all
exit
.endc
.end
