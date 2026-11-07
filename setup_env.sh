#!/bin/bash
set -euo pipefail

# Clone Chloe Li's upstream repository alongside our scripts if not already cloned
if [ ! -d "model_spec_midtraining" ]; then
    git clone --recurse-submodules https://github.com/chloeli-15/model_spec_midtraining.git
fi

pip install -r requirements.txt
pip install -e model_spec_midtraining/
pip install -e model_spec_midtraining/safety-tooling/
echo "Environment setup complete!"
