---
title: P-013 grid screening
emoji: ⚡
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: app.py
license: mit
short_description: AI ranks 80 line failures, the real solver checks them
---

# If one power line fails, does anything overload?

An AI ranks all 80 possible single line failures on the IEEE 57-bus grid, and
the real Newton-Raphson solver (pandapower) checks only the risky ones.

- Recorded results for 175 test days: https://hamza-atiq.github.io/p013-grid-contingency-screening/
- Code, tests and the full write-up: https://github.com/Hamza-Atiq/p013-grid-contingency-screening

The grid data is `pglib_opf_case57_ieee.m` from the IEEE PES Power Grid Library
(PGLib-OPF), CC BY 4.0.
