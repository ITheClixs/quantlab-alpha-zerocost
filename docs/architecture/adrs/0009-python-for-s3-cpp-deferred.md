# ADR 0009: Python for S3 (feeds + brokers); C++ adapters deferred until measured benefit

## Status
Accepted, 2026-05-17.

## Context
C++ is the conventional language for low-latency quant infrastructure. The operator
asked whether S3 (real-time feeds + broker abstraction) should be written in C++ to
match industry practice.

## Decision
S3 is implemented in Python. C++ adapters are deferred until profiling on real
production traffic shows a measurable latency win.

## Latency math at decision time
- Hardware: MacBook Air M4 in Istanbul (not a colo cabinet).
- Feeds: Binance / Coinbase public WebSocket, network RTT ~50-200 ms.
- Event rate at peak: Binance BTCUSDT aggTrade ~500-2 000 events/sec.
- Python json.loads on each event: ~5 microseconds.

A C++ feed handler would save ~5 microseconds per event. The network RTT (~50-100 ms)
is 20 000x larger. The latency win is unmeasurable on this network path.

## Architectural enabler
The FeedAdapter Protocol boundary is language-agnostic. A future C++ adapter wrapped
via pybind11 can drop in behind the same Protocol with zero downstream changes to
S1 / S2 / S4 / backtester.

## Triggers that would justify revisiting
- Co-location with the exchange (network RTT drops to single-digit microseconds).
- Migration to ITCH / OUCH / FIX-FAST binary feeds (parsing becomes the bottleneck).
- Aggregating L2 / L3 order books across >= 5 venues simultaneously.
- Production profiling shows p99 event-handler latency > 100 ms attributable to Python.

## Consequences
+ Faster iteration: hot-reload strategy modules in seconds.
+ Smaller build surface: no cmake / vcpkg / pybind11 in the default install path.
+ The heavy compute (Polars, NumPy, LightGBM, llama.cpp) is already C++ underneath;
  Python is the orchestration layer where its speed cost is invisible.
- A future migration to C++ for a specific bottleneck is non-trivial (pybind11 wrappers,
  separate test surface). Mitigated by keeping the Protocol boundary clean.
