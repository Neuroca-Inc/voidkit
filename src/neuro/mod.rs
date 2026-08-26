/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
*/

//! Neural plasticity and learning functions.
//!
//! This module contains advanced neural plasticity algorithms including
//! stabilized reward functions and STDP modulation.

pub mod advanced_sie;

pub use advanced_sie::{apply_quadratic_stdp_modulation, calculate_stabilized_reward};
