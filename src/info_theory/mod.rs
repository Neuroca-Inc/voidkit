/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

Information theory module - entropy, mutual information, KL divergence.
*/

pub mod information_theory;

pub use information_theory::{
    calculate_entropy, 
    calculate_mutual_information, 
    calculate_kl_divergence,
    information_bottleneck
};
