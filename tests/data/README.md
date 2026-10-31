# tests/data

Saved simulator output used as parser fixtures.

- `xy_block_mismatch_off_one_sample.res` (TASK-435): the header, first sample
  row and trailer of the `.res` file Xyce 7.10 writes for
  `docs/archive/reviews/monte-carlo-sky130/design-probes-2026-10-02/xy_block_mismatch_off.cir`,
  run unchanged with `Xyce -randseed 7 xy_block_mismatch_off.cir` on
  2026-10-02 (sky130A `tt_mm`, `PDK_ROOT` set to a sky130 install). The saved probe
  transcript keeps only the header; its header line is identical to this
  file's. Values are Xyce's normalised draws (z); row 0's
  `XOFF:XM1:...:VTH0` is 2.01835591.
- `ng_showmod_sample.log` (TASK-429): the ngspice-42 `-o` log of one
  `record_draws=True` sample of the TASK-300 diff pair (`nmos1p8` W=5 L=2,
  `tt_mm`, sample 0 of seed 0), written by silk on 2026-10-03 with
  `PDK_ROOT` set to a sky130 install; only the sample folder path is replaced by
  `<sample dir>`. Unlike the probe transcript, the `-o` log puts a blank line
  after each `model` header. `dut.m1`'s vth0 is 0.522291.
