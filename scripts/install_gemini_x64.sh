#!/usr/bin/env bash
# Install Ultraleap Gemini (v5) hand tracking + LeapC on Ubuntu 22.04 x86_64.
# Source: https://docs.ultraleap.com/linux/
set -euo pipefail

if [ "$(uname -m)" != "x86_64" ]; then
  echo "This script is for x86_64 only. For ARM boards see README (ARM section)." >&2
  exit 1
fi

# 1. Ultraleap signing key
wget -qO - https://repo.ultraleap.com/keys/apt/gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/ultraleap.gpg > /dev/null

# 2. Ultraleap apt repository
echo 'deb [arch=amd64] https://repo.ultraleap.com/apt stable main' | sudo tee /etc/apt/sources.list.d/ultraleap.list

# 3. Install service + control panel + OpenXR layer (meta package)
sudo apt update
sudo apt install -y ultraleap-hand-tracking

# 4. Accept the EULA if the installer did not prompt for it
leapctl eula || true

# 5. Build tools for the Python bindings
sudo apt install -y build-essential python3-dev python3-venv portaudio19-dev

echo
echo "Done. Plug in the Leap Motion Controller, then run:"
echo "  ultraleap-hand-tracking-control-panel   # check the camera view and tracked hands"
echo "  leapctl --help                          # CLI for the same settings"
echo
echo "LeapC files should now be in:"
echo "  /usr/include/LeapC.h"
echo "  /usr/lib/ultraleap-hand-tracking-service/libLeapC.so"
ls -l /usr/include/LeapC.h /usr/lib/ultraleap-hand-tracking-service/libLeapC.so 2>/dev/null || \
  echo "  (not found yet; check the package installed correctly)"
