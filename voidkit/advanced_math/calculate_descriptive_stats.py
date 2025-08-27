"""
Copyright © 2025 Justin K. Lietz, Neuroca, Inc. All Rights Reserved.

This research is protected under a dual-license to foster open academic
research while ensuring commercial applications are aligned with the project's ethical principles. Commercial use requires written permission from Justin K. Lietz. 
See LICENSE file for full terms.
"""

import numpy as np
from scipy import stats

def calculate_descriptive_stats(data, nan_policy='propagate', ddof=0):
    """
    Calculate descriptive statistics for a given dataset.
    
    Parameters
    ----------
    data : array_like
        Input data array. Must be convertible to a 1D NumPy array of numbers.
    nan_policy : {'propagate', 'omit', 'raise'}, optional
        Defines how to handle NaN values:
        - 'propagate': returns nan for statistics when NaN values are present (default)
        - 'omit': ignores NaN values when computing statistics
        - 'raise': raises an error if NaN values are present
    ddof : int, optional
        Delta Degrees of Freedom for standard deviation and variance calculations (default: 0)
    
    Returns
    -------
    dict
        Dictionary containing the following descriptive statistics:
        - 'count': Number of observations
        - 'mean': Arithmetic mean
        - 'median': Median value
        - 'std': Standard deviation
        - 'var': Variance
        - 'min': Minimum value
        - 'max': Maximum value
        - 'q1': First quartile (25th percentile)
        - 'q3': Third quartile (75th percentile)
        - 'iqr': Interquartile range (q3 - q1)
    
    Raises
    ------
    TypeError
        If input data cannot be converted to a numeric array.
    ValueError
        If nan_policy is not one of {'propagate', 'omit', 'raise'}.
        If input data is empty.
        If NaN values are present and nan_policy is 'raise'.
    
    Examples
    --------
    >>> import numpy as np
    >>> data = [1, 2, 3, 4, 5]
    >>> stats = calculate_descriptive_stats(data)
    >>> print(stats['mean'])
    3.0
    >>> print(stats['std'])
    1.5811388300841898
    """
    # Validate nan_policy parameter
    valid_policies = ['propagate', 'omit', 'raise']
    if nan_policy not in valid_policies:
        raise ValueError(f"nan_policy must be one of {valid_policies}, got '{nan_policy}'")
    
    # Check if data is None
    if data is None:
        raise ValueError("Input data cannot be None")
    
    # Convert input to numpy array
    try:
        data_array = np.asarray(data, dtype=float)
    except (ValueError, TypeError) as e:
        raise TypeError(f"Failed to convert input data to numeric array: {str(e)}")
    
    # Ensure data is 1D
    if data_array.ndim > 1:
        raise ValueError(f"Input data must be 1-dimensional, got {data_array.ndim}-dimensional data")
    
    # Check if data is empty
    if data_array.size == 0:
        raise ValueError("Input data cannot be empty")
    
    # Handle NaN values according to policy
    if nan_policy == 'raise' and np.isnan(data_array).any():
        raise ValueError("Input data contains NaN values")
    elif nan_policy == 'omit':
        data_array = data_array[~np.isnan(data_array)]
        if data_array.size == 0:
            raise ValueError("No valid data remaining after removing NaN values")
    
    # Calculate basic statistics
    count = len(data_array)
    mean = np.mean(data_array)
    median = np.median(data_array)
    
    # Calculate variance and standard deviation with ddof
    var = np.var(data_array, ddof=ddof)
    std = np.sqrt(var)
    
    # Calculate min and max
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    # Calculate quartiles and IQR
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    
    return {
        'count': count,
        'mean': mean,
        'median': median,
        'var': var,
        'std': std,
        'min': min_val,
        'max': max_val,
        'q1': q1,
        'q3': q3,
        'iqr': iqr
    }


# Alias for the main function to match expected API
descriptive_stats = calculate_descriptive_stats


def main(argv=None):
    """CLI entry point for voidkit-stats command."""
    import argparse
    import json
    import sys
    
    if argv is None:
        argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        prog="voidkit-stats",
        description="Calculate descriptive statistics for numeric data (VDM Advanced Math)"
    )
    parser.add_argument(
        "data",
        nargs="*",
        type=float,
        help="Numeric values to analyze (space-separated)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--nan-policy",
        choices=["propagate", "omit", "raise"],
        default="propagate",
        help="How to handle NaN values (default: propagate)"
    )
    
    args = parser.parse_args(argv)
    
    if not args.data:
        parser.error("No data provided")
    
    try:
        stats = descriptive_stats(args.data, nan_policy=args.nan_policy)
        
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Descriptive Statistics:")
            print(f"Count: {stats['count']}")
            print(f"Mean: {stats['mean']:.6f}")
            print(f"Median: {stats['median']:.6f}")
            print(f"Standard Deviation: {stats['std']:.6f}")
            print(f"Variance: {stats['var']:.6f}")
            print(f"Minimum: {stats['min']:.6f}")
            print(f"Maximum: {stats['max']:.6f}")
            print(f"Q1 (25th percentile): {stats['q1']:.6f}")
            print(f"Q3 (75th percentile): {stats['q3']:.6f}")
            print(f"IQR: {stats['iqr']:.6f}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
