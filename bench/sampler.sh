#!/usr/bin/env bash
# Utilization sampler without sudo: GPU device utilization (ioreg), CPU% and RSS of HerculeX python processes, 1 Hz.
out=$1
echo -e "t\tgpu_util\tgpu_mem_MB\tcpu_main\tcpu_workers\tcpu_total\trss_GB\tfree_GB" >"$out"
while :; do
    g=$(ioreg -r -d 1 -w 0 -c IOAccelerator 2>/dev/null | grep -o '"PerformanceStatistics" = {[^}]*}' | head -1)
    gu=$(echo "$g" | grep -o '"Device Utilization %"=[0-9]*' | cut -d= -f2)
    gm=$(echo "$g" | grep -o '"In use system memory"=[0-9]*' | cut -d= -f2)
    read -r cm cw rss <<<"$(ps -axo pcpu=,rss=,command= | awk '/runtime\/python/ && !/awk/{ if ($0 ~ /multiprocessing/) w+=$1; else m+=$1; r+=$2 } END{printf "%.0f %.0f %.2f", m, w, r/1048576}')"
    ct=$(ps -A -o pcpu= | awk '{s+=$1} END{printf "%.0f", s}')
    echo -e "$(date +%s)\t${gu:-0}\t$((${gm:-0} / 1048576))\t$cm\t$cw\t$ct\t$rss\t$(df -g ~ | awk 'NR==2{print $4}')" >>"$out"
    sleep 1
done
