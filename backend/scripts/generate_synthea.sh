#!/usr/bin/env bash
set -euo pipefail

PATIENT_COUNT="${1:-500}"
OUTPUT_DIR="$(pwd)/data/synthea"
SYNTHEA_DIR="$(pwd)/data/synthea-src"

echo "Generating ${PATIENT_COUNT} synthetic FHIR R4 patients..."

if [ ! -d "$SYNTHEA_DIR" ]; then
  git clone https://github.com/synthetichealth/synthea "$SYNTHEA_DIR"
fi

cd "$SYNTHEA_DIR"
./gradlew build -x test --quiet

./run_synthea \
  -p "$PATIENT_COUNT" \
  --exporter.fhir.export=true \
  --exporter.fhir.transaction_bundle=false \
  --exporter.baseDirectory="$OUTPUT_DIR" \
  Massachusetts

echo "Done. FHIR bundles written to $OUTPUT_DIR/fhir/"
