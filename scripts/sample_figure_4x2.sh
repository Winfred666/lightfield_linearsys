#!/bin/bash

# Check input
if [ -z "$1" ]; then
    echo "Usage: $0 <loss_curve_directory>"
    exit 1
fi

INPUT_DIR="$1"
# Remove trailing slash if present
INPUT_DIR="${INPUT_DIR%/}"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory $INPUT_DIR does not exist."
    exit 1
fi

# Parent directory
PARENT_DIR="$(dirname "$INPUT_DIR")"
BASENAME="$(basename "$INPUT_DIR")"
OUTPUT_DIR="${PARENT_DIR}/${BASENAME}_grid"

mkdir -p "$OUTPUT_DIR"

echo "Scanning $INPUT_DIR..."

# Get list of PNG files, naturally sorted
# We use find to be safe, then sort.
mapfile -t FILES < <(find "$INPUT_DIR" -maxdepth 1 -name "*.png" | sort -V)

COUNT=${#FILES[@]}
echo "Found $COUNT images."

if [ "$COUNT" -lt 8 ]; then
    echo "Warning: Less than 8 images found. Cannot do 4x2 sampling properly."
    if [ "$COUNT" -eq 0 ]; then
        exit 1
    fi
fi

# Select 8 indices
INDICES=()
if [ "$COUNT" -eq 1 ]; then
    # Edge case 1 file
    for i in {1..8}; do INDICES+=(0); done
else
    # spacing = (N - 1) / 7
    # We want 0, ..., N-1
    for i in {0..7}; do
        # Floating point calc for index
        # idx = round( i * (COUNT - 1) / 7.0 )
        idx=$(python3 -c "print(int(round($i * ($COUNT - 1) / 7.0)))")
        INDICES+=("$idx")
    done
fi

echo "Selected indices: ${INDICES[*]}"

# Build ffmpeg input args
FFMPEG_INPUTS=""
CHOSEN_FILES=()

for idx in "${INDICES[@]}"; do
    # Modulo arithmetic in case indices go OOB (shouldn't with the logic above)
    safe_idx=$((idx % COUNT))
    f="${FILES[$safe_idx]}"
    FFMPEG_INPUTS="$FFMPEG_INPUTS -i $f"
    CHOSEN_FILES+=("$f")
done

# Build filter complex
# 4x2 grid
# Row 1: 0,1,2,3
# Row 2: 4,5,6,7
FILTER_COMPLEX=" \
[0][1][2][3]hstack=inputs=4[row1]; \
[4][5][6][7]hstack=inputs=4[row2]; \
[row1][row2]vstack=inputs=2[out]"

OUTPUT_FILE="${OUTPUT_DIR}/sampled_4x2.png"

echo "Generating $OUTPUT_FILE..."

# Overwrite (-y)
ffmpeg $FFMPEG_INPUTS \
    -filter_complex "$FILTER_COMPLEX" \
    -map "[out]" \
    -y "$OUTPUT_FILE" 2>/dev/null

if [ $? -eq 0 ]; then
    echo "Success! Created $OUTPUT_FILE"
else
    echo "FFmpeg failed."
    # Run again without stderr redirection to show error
    ffmpeg $FFMPEG_INPUTS \
        -filter_complex "$FILTER_COMPLEX" \
        -map "[out]" \
        -y "$OUTPUT_FILE"
fi
