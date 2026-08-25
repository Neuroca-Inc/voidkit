# Naming Corrections

Corrections in this cleanup are documentation/interface corrections only. They do not alter equations, algorithms, validation claims, or numerical behavior.

## SIE acronym expansion

**File:** `src/neuro/advanced_sie.rs`

**Historical text:** `Advanced Stabilized Information-theoretic Engagement (SIE) functions.`

**Corrected text:** `Self-Improvement Engine (SIE) multi-objective reward functions.`

**Reason:** SIE means **Self-Improvement Engine**. The module implementation itself already identifies its main calculation as a stabilized multi-objective reward function combining temporal-difference error, novelty, habituation, self-benefit, and external-reward-dependent weighting.

**Scientific effect:** none. Only Rust module-level documentation was changed.

**Original source SHA-256:** `b4d7b88ca4571466e979cce82f7a78b5cf1464a9ca87f1d99b87a2a02be7393a`
