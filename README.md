# MIDI R1

A transformer-based model for MIDI sequence modeling using the DeepSeek V3 simplified architecture and the 
GigaMIDI dataset.

## Setup

```bash
# Clone the repo
git clone <repo-url> && cd midi_r1

# Create a virtual environment
python -m venv .venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode so that `midiR1` is discoverable
pip install -e .