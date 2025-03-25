#!/usr/bin/env python3
import yaml
import os
import sys
from collections import defaultdict

def count_parameters(yaml_file):
    """
    Count the total number of parameters in a model configuration file.
    """
    # Read the YAML file
    with open(yaml_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return
    
    # Check if 'param' exists in the config
    if 'param' not in config:
        print("No 'param' section found in the configuration file.")
        return
    
    # Initialize counters
    total_params = 0
    param_counts_by_type = defaultdict(int)
    
    # Iterate through each parameter entry
    for param in config['param']:
        # Extract parameter dimensions
        name = param['name']
        rown = param['rown']
        coln = param['coln']
        
        # Calculate number of parameters for this entry
        # For standard 2D parameters: rown * coln
        # For rank-1 parameters: use rown (assuming it's a vector)
        if 'rank' in param and param['rank'] == 1:
            param_count = rown
        else:
            param_count = rown * coln
        
        # Add to total
        total_params += param_count
        
        # Categorize by parameter type/layer
        # Extracting the base name (before any /)
        base_name = name.split('/')[0].split('_')[0] if '/' in name else name.split('_')[0]
        param_counts_by_type[base_name] += param_count
        
        # Print individual parameter details
        print(f"{name}: {param_count:,} parameters ({rown} × {coln})")
    
    # Print summary statistics
    print("\n" + "="*60)
    print(f"Total parameters: {total_params:,}")
    print("="*60)
    
    # Print breakdown by parameter type/layer
    print("\nParameters by type:")
    for type_name, count in sorted(param_counts_by_type.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_params) * 100
        print(f"{type_name}: {count:,} ({percentage:.2f}%)")

if __name__ == "__main__":
    # Use command line argument if provided, otherwise use default path
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
    else:
        raise ValueError("No YAML file provided")
    
    if not os.path.exists(yaml_file):
        print(f"File not found: {yaml_file}")
        sys.exit(1)
    
    count_parameters(yaml_file) 