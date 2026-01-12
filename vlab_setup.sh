#!/bin/bash

# VLAB Environment Setup Script
# Run this script to set up VLAB environment variables and PATH

echo "Setting up VLAB environment..."

# Change to vlab directory
cd /app/vlab-5.0

# Run VLAB post-installation if not already done
if [ ! -f "bin/sourceme.sh" ]; then
    echo "Running VLAB post-installation setup..."
    ./bin/postinstall.sh
fi

# Source the VLAB environment
if [ -f "bin/sourceme.sh" ]; then
    echo "Sourcing VLAB environment from $(pwd)/bin/sourceme.sh"
    source bin/sourceme.sh
    
    # Export environment for current session
    export VLABROOT VLABBIN PATH VLABDOCDIR VLABOBJECTBIN VLABDAEMONBIN
    export VLABCONFIGDIR VLABHBROWSERBIN VLABBROWSERBIN LPFGPATH LPFGRESOURCES
    export VVDIR VVEDIR LD_LIBRARY_PATH VLABTMPDIR
    
    echo "VLAB environment loaded successfully!"
    echo "VLABROOT: $VLABROOT"
    echo "VLABBIN: $VLABBIN"
    echo "VLAB commands should now be available in PATH"
    
    # Test if VLAB commands are available
    if command -v browser >/dev/null 2>&1; then
        echo "✓ VLAB commands are available!"
        echo "You can now run VLAB tools like: browser, lpfg, cpfg, etc."
    else
        echo "⚠ VLAB commands not found in PATH. Something may be wrong."
    fi
else
    echo "ERROR: bin/sourceme.sh not found. VLAB setup failed."
    exit 1
fi

# Initialize VLAB configuration if not already done
echo "Initializing VLAB configuration..."
./bin/govlab.sh

echo "VLAB setup complete!"