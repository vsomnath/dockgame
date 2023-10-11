#!/bin/bash

DATA_DIR=${1:-"data"}
ZENODO_ID="8408573"
BASE_URL="https://zenodo.org/record/${ZENODO_ID}/files"

mkdir -p "${DATA_DIR}/raw"
cd "${DATA_DIR}/raw"

for filename in "db5_raw.tar.gz" "dips_raw.tar.gz"
do  
    echo "Downloading ${filename} to ${DATA_DIR}/raw"
    curl -L -o "${filename}" "${BASE_URL}/${filename}?download=1"
    tar -xvf ${filename}
    rm ${filename}
done

echo "Downloaded data saved to ${DATA_DIR}"
