#!/usr/bin/env bash
set -e

VENDOR_DIR="vendor/deca"

if [ ! -d "$VENDOR_DIR" ]; then
  echo "Cloning DECA into $VENDOR_DIR..."
  git clone --depth 1 https://github.com/YadiraF/DECA.git "$VENDOR_DIR"

  # Add a minimal setup.py so we can pip-install it
  cat > "$VENDOR_DIR/setup.py" << 'EOF'
from setuptools import setup, find_packages
setup(
  name="deca",
  version="0.1",
  packages=find_packages(),
)
EOF

  echo "Installing DECA in editable mode..."
  python3 -m pip install -e "$VENDOR_DIR"
else
  echo "✅ DECA already present in $VENDOR_DIR"
fi
