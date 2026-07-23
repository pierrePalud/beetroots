#!/bin/bash

# Define the grid values for your parameters
values_a=("1.1" "1.5" "2.0")  # sigma_m_float_linscale
values_b=("1.38715e-10" "2.05e-11" "1.92e-9")  # sigma_a

ORIGINAL_YAML="./data/toycases/input_nn_bo_toycase.yaml"
TEMP_YAML="./data/toycases/input_temp.yaml"

# Iterate over parameter A (sigma_m_float_linscale)
for a in "${values_a[@]}"; do
    for b in "${values_b[@]}"; do
        
        # 1. Format sigma_m (1.1 -> 1p1)
        a_tag=$(echo "$a" | tr '.' 'p')
        
        # 2. Format sigma_a (1.38715e-10 -> 1p3e10)
        # First, get 1.4e-10 format
        b_raw=$(LC_ALL=C printf "%.1e" "$b")
        # Step B: 1.4e-10 -> 1p4e10 (replace dot, remove + and -)
        b_tag=$(echo "$b_raw" | tr '.' 'p' | sed 's/[+-]//g')
        echo "$b_raw"
        echo "$b_tag"
        
        # 3. Construct the full simu_name
        NEW_NAME="approx_optim_nn_bo_grid_m${a_tag}_a${b_tag}_${CURRENT_DATE}"
        
        echo "----------------------------------------------------------"
        echo "Running: $NEW_NAME"
        
        # Setup the temporary YAML
        cp "$ORIGINAL_YAML" "$TEMP_YAML"
        
        # Update sigma_m_float_linscale in the file (keep original high precision)
        sed -i "s/^sigma_m_float_linscale:.*/sigma_m_float_linscale: $a/" "$TEMP_YAML"
        
        # Update sigma_a in the file
        sed -i "s/^sigma_a:.*/sigma_a: $b/" "$TEMP_YAML"
        
        # Update simu_name inside simu_init
        sed -i "s/simu_name: \".*\"/simu_name: \"$NEW_NAME\"/" "$TEMP_YAML"
        
        # Execute
        uv run python src/beetroots/approx_optim/nn_bo_toycase.py \
            input_temp.yaml ./data/toycases/ ./data/models ./old_outputs
            
    done
done

rm "$TEMP_YAML"