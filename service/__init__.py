"""DFS worker service package.

A FastAPI application that serves the light, interactive numeric API (warm,
sim arrays held in memory) and — in later phases — runs the heavy Stage B/C
pipeline as background jobs. It reuses the existing repo modules
(``stage_d``, ``mlb_lineup_builder``, ``portfolio``, ``field_simulator`` …)
with no rewrite of the numeric logic. See ``ARCHITECTURE.md``.
"""
