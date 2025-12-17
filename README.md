eTSM-PSS7

eTSM-PSS7 is a research prototype for evaluating self-maintenance in large language models.
It introduces an external persona and memory exoskeleton with multi-timescale updates, enabling quantitative measurement of long-horizon consistency, contradiction resilience, and persona stability without modifying model weights.

This repository focuses on evaluation protocols and metrics, not on releasing a full reference implementation.
What This Repository Provides

This repository contains:

A formalized evaluation framework for long-horizon self-maintenance in LLM-based agents

Clearly defined metrics for measuring persona stability and contradiction resilience

A transparent research positioning separating design exploration from experimental evidence

It does not claim AGI, consciousness, or sentient behavior.
Evaluation Metrics

The following metrics are used to evaluate self-maintenance behavior:

Contradiction Rate
Frequency at which an agent violates previously stated policies or values under contradiction or noise injection.

Policy Flip Count
Number of observable reversals in explicit commitments across long-horizon interactions.

Persona Drift (v_t stability)
Norm-based deviation of the persona vector over time.

Memory Consistency
Degree to which retrieved long-term memories remain aligned with current persona and stated intentions.

These metrics are designed to be model-agnostic and reproducible.
Evaluation Protocol

The evaluation protocol is structured around:

Long-horizon interaction sequences

Explicit contradiction and noise injection

Fixed measurement intervals for persona and memory states

Separation of fast inference, slow persona updates, and ultra-slow adaptation

The protocol definition is intentionally independent of any specific LLM or vendor.
Transparency Statement

Earlier multi-model dialogue logs referenced in related notes or articles are treated as prompt-guided simulations for design-space exploration, not as experimental proof.

Scientific claims are grounded only in reproducible evaluation protocols and measurable outcomes.

Code Availability Policy

A reference implementation exists and is currently evaluated under controlled conditions.

To avoid premature commoditization and to preserve research integrity, executable code is not publicly released at this stage.
Code may be shared selectively for collaborative research or under appropriate agreements.

Scope and Limitations

This work evaluates self-maintenance behavior, not general intelligence.

No claims are made regarding autonomy, consciousness, or agency.

Results depend on the quality of the underlying language model and scoring functions.

Intended Audience

This project is intended for researchers and engineers working on:

LLM-based agents

Long-term memory and persona stability

Alignment and evaluation methodologies

Multi-timescale learning systems
