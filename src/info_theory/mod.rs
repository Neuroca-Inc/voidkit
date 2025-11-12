/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Information theory module - entropy, mutual information, KL divergence.
*/

pub mod information_theory;

pub use information_theory::{
    calculate_entropy, calculate_kl_divergence, calculate_mutual_information,
    information_bottleneck,
};
