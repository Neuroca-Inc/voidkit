/*
Copyright © 2025 Justin K. Lietz, Neuroca, Inc.
SPDX-License-Identifier: BSD-3-Clause

Licensed under the BSD 3-Clause License. See LICENSE in the repository root.
*/

//! Spatial Hash Grid Data Structure
//!
//! Efficient spatial partitioning for collision detection and nearest neighbor queries.

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::collections::HashMap;

/// Spatial hash grid for efficient spatial queries.
///
/// Partitions space into uniform grid cells for O(1) insertion and
/// O(k) query where k is the number of objects in nearby cells.
#[pyclass]
pub struct SpatialHashGrid {
    cell_size: f64,
    grid: HashMap<(i64, i64), Vec<usize>>, // Maps cell coords to object indices
    objects: Vec<(f64, f64)>,              // Stores object positions
}

#[pymethods]
impl SpatialHashGrid {
    /// Create a new spatial hash grid.
    ///
    /// # Arguments
    /// * `cell_size` - Size of each grid cell
    #[new]
    pub fn new(cell_size: f64) -> PyResult<Self> {
        if cell_size <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "cell_size must be positive",
            ));
        }

        Ok(SpatialHashGrid {
            cell_size,
            grid: HashMap::new(),
            objects: Vec::new(),
        })
    }

    /// Insert an object at a given point.
    ///
    /// # Arguments
    /// * `point` - 2D position [x, y]
    ///
    /// # Returns
    /// Index of the inserted object
    pub fn insert(&mut self, point: PyReadonlyArray1<f64>) -> PyResult<usize> {
        let point_arr = point.as_array();

        if point_arr.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "point must be a 2D array [x, y]",
            ));
        }

        let x = point_arr[0];
        let y = point_arr[1];
        let cell = self.hash_point(x, y);

        // Add object
        let obj_idx = self.objects.len();
        self.objects.push((x, y));

        // Add to grid
        self.grid.entry(cell).or_default().push(obj_idx);

        Ok(obj_idx)
    }

    /// Query objects within a radius of a point.
    ///
    /// # Arguments
    /// * `point` - 2D position [x, y]
    /// * `radius` - Search radius
    ///
    /// # Returns
    /// List of object indices within the radius
    pub fn query(&self, point: PyReadonlyArray1<f64>, radius: f64) -> PyResult<Vec<usize>> {
        let point_arr = point.as_array();

        if point_arr.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "point must be a 2D array [x, y]",
            ));
        }

        let x = point_arr[0];
        let y = point_arr[1];

        // Calculate cell bounds for the search radius
        let min_cell = self.hash_point(x - radius, y - radius);
        let max_cell = self.hash_point(x + radius, y + radius);

        let mut results = Vec::new();

        // Check all cells in the bounding box
        for i in min_cell.0..=max_cell.0 {
            for j in min_cell.1..=max_cell.1 {
                if let Some(cell_objects) = self.grid.get(&(i, j)) {
                    for &obj_idx in cell_objects {
                        let (obj_x, obj_y) = self.objects[obj_idx];
                        let dx = obj_x - x;
                        let dy = obj_y - y;
                        let dist_sq = dx * dx + dy * dy;

                        if dist_sq <= radius * radius {
                            results.push(obj_idx);
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get all objects in the same cell as the given point.
    ///
    /// # Arguments
    /// * `point` - 2D position [x, y]
    ///
    /// # Returns
    /// List of object indices in the same cell
    pub fn get_collisions(&self, point: PyReadonlyArray1<f64>) -> PyResult<Vec<usize>> {
        let point_arr = point.as_array();

        if point_arr.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "point must be a 2D array [x, y]",
            ));
        }

        let x = point_arr[0];
        let y = point_arr[1];
        let cell = self.hash_point(x, y);

        Ok(self.grid.get(&cell).cloned().unwrap_or_default())
    }

    /// Clear all objects from the grid.
    pub fn clear(&mut self) {
        self.grid.clear();
        self.objects.clear();
    }

    /// Get the number of objects in the grid.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Return whether the grid contains no objects.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

impl SpatialHashGrid {
    /// Hash a 2D point to a grid cell coordinate.
    fn hash_point(&self, x: f64, y: f64) -> (i64, i64) {
        (
            (x / self.cell_size).floor() as i64,
            (y / self.cell_size).floor() as i64,
        )
    }
}
