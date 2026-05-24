# GPU SpMV: Read the project as an engineering artifact

<CalloutPanel title="Project Positioning" tone="success">
This site is written for interviewers, open-source readers, and performance engineers. The whitepaper landing page leads with conclusions, then points to the design decisions and evidence chain behind them.
</CalloutPanel>

## Why this project deserves a whitepaper

- SpMV is a classic **memory-bandwidth-bound** workload, so performance depends more on access patterns than raw arithmetic throughput.
- The interesting part is not only which kernel exists, but **why it is chosen, when it is chosen, and how that choice is justified**.
- This project combines CUDA performance work with RAII resource management, explicit error handling, spec-driven development, and readable documentation.

## What this whitepaper is meant to answer

1. Why the problem matters and where the real bottlenecks are.
2. What each optimized kernel and the selector are responsible for.
3. How performance, engineering discipline, and explainability are tied together.
4. Where to continue reading for architecture, API usage, performance interpretation, and references.

## Reading Path

| Page | Role |
|:-----|:-----|
| [Design Philosophy](/en/whitepaper/philosophy) | See the architectural priorities and trade-offs |
| [Performance Analysis](/en/whitepaper/performance) | Learn how to interpret the benchmark evidence |
| [Architecture Overview](/en/architecture/overview) | Understand the execution pipeline and module boundaries |
| [API Reference](/en/api/spmv) | Inspect the external interface |
| [References](/en/references) | Review papers, projects, and further reading |
