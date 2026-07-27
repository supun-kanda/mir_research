#!/usr/bin/env bash
# Clone the ASAP dataset (scores, MIDI performances, beat annotations).
# Small (no audio) -- a few hundred MB. Audio has to be built separately
# from MAESTRO; see download_maestro.sh and build_asap_audio.py.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEST="data/asap-dataset"
REPO="https://github.com/fosfrancesco/asap-dataset.git"

if [ -d "$DEST/.git" ]; then
    echo "$DEST already exists, pulling latest instead of re-cloning."
    git -C "$DEST" pull
else
    git clone "$REPO" "$DEST"
fi

echo
echo "ASAP dataset ready at $DEST"
echo "It has scores/MIDI/annotations but no audio yet."
echo "Next: ./scripts/download_maestro.sh, then python scripts/build_asap_audio.py"
