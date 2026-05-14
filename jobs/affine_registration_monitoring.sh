OUTDIR=/home/ids/gmargari-24/Data/Affine_registered
LOG=register_affine_parallel_785014.out

DONE=$(ls "$OUTDIR"/*_R.nii.gz 2>/dev/null | wc -l)
FIRST=$(ls -tr "$OUTDIR"/*_R.nii.gz 2>/dev/null | head -1)
LAST=$(ls -t "$OUTDIR"/*_R.nii.gz 2>/dev/null | head -1)

if [ "$DONE" -gt 0 ]; then
    FIRST_TIME=$(stat -c %Y "$FIRST")
    LAST_TIME=$(stat -c %Y "$LAST")
    WALL_SECS=$((LAST_TIME - FIRST_TIME))
    echo "Completed: $DONE files"
    echo "Total time (first-to-last output): $((WALL_SECS/60)) min $((WALL_SECS%60)) sec"
    echo ""
fi

printf "%-4s  %-28s  %10s  %14s\n" "#" "FILE" "ELAPSED"  "MEMORY"
echo "------------------------------------------------------------"

paste \
  <(grep "DONE" "$LOG" | grep -oP 'file\s*=\s*\K\S+') \
  <(grep "DONE" "$LOG" | grep -oP 'elapsed=\s*\K[0-9.]+') \
  <(grep "DONE" "$LOG" | grep -oP 'peak_mem=\s*\K\S+') | \
  sort -k1 | \
  awk '{ n++; printf "%-4d  %-28s  %7.1fs (%4.1fm)  %8s\n", n, $1, $2, $2/60, $3 }'

echo ""
grep "DONE" "$LOG" | grep -oP 'elapsed=\s*\K[0-9.]+' | \
  awk '{sum+=$1; n++} END {if(n>0) printf "Avg per file: %.1f min\n", sum/n/60}'