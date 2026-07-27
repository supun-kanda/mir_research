#!/usr/bin/env bash
# Download MAESTRO v2.0.0 (full audio+MIDI). ASAP's metadata.csv was built
# against this exact version's file layout, so stick with v2.0.0 rather
# than v3.0.0 -- v3 renamed/removed a handful of recordings and would break
# some of the maestro_audio_performance path lookups.
#
# THIS IS A ~103GB DOWNLOAD. Not run automatically -- run it yourself when
# you actually want the audio, e.g.:
#   ./scripts/download_maestro.sh data/maestro-v2.0.0
set -euo pipefail

DEST_DIR="${1:-data/maestro-v2.0.0}"
URL="https://storage.googleapis.com/magentadata/datasets/maestro/v2.0.0/maestro-v2.0.0.zip"

echo "About to download MAESTRO v2.0.0 (~103GB compressed, ~122GB unzipped) to:"
echo "  $DEST_DIR"
echo "URL: $URL"
read -p "Continue? [y/N] " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 1
fi

mkdir -p "$DEST_DIR"
ZIP_PATH="$DEST_DIR/maestro-v2.0.0.zip"

curl -L --continue-at - -o "$ZIP_PATH" "$URL"

echo "Unzipping..."
unzip -q "$ZIP_PATH" -d "$DEST_DIR"

echo
echo "Done. MAESTRO audio is under $DEST_DIR/maestro-v2.0.0/<year>/*.wav"
echo "Next: python scripts/build_asap_audio.py --maestro $DEST_DIR/maestro-v2.0.0"
echo "You can delete the zip afterwards to reclaim ~103GB: rm $ZIP_PATH"
