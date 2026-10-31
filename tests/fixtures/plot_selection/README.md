# Same-kind plot selection evidence

Real measurements produced by ngspice-42 on 2026-10-20 using
`two_transients.cir`. The deck runs a 1 V transient, changes the source to
9 V, then runs another transient. Both plots have the saved header
`Transient Analysis`; their internal tran1/tran2 names are absent.

`combined.raw` saves tran1 then tran2 (1 V, 9 V). `reversed.raw` saves
tran2 then tran1 (9 V, 1 V). These are the original ASCII measurement
files, retained in both write orders. Regression tests read them without
launching a simulator. They live separately from the viewer fixture corpus.
