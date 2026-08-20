#!/usr/bin/env bash
set -a
# shellcheck disable=SC1091
source "$(dirname "$0")/../.env"
set +a

N_START=0      # start index
N=40000        # max total points
P=30           # number of processes
CHUNK=$((N / P))

for ((i=0; i<P; i++)); do
    START=$((i * CHUNK + N_START))
    END=$(( (i+1) * CHUNK + N_START))

    # last chunk takes the remainder
    if [ $i -eq $((P-1)) ]; then
        END=$((N + N_START))
    fi

    echo "Launching $START -> $END"
    python download_gee_data.py --start $START --stop $END &
done

wait
echo "All workers finished."
