/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles.
Commercial use requires written permission from Justin K. Lietz.
See LICENSE file for full terms.
*/

//! Neural plasticity and learning functions.
//!
//! This module contains advanced neural plasticity algorithms including
//! stabilized reward functions and STDP modulation.

pub mod advanced_sie;

pub use advanced_sie::{calculate_stabilized_reward, apply_quadratic_stdp_modulation};
