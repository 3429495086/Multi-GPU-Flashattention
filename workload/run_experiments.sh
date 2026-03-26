DEMO=./demo

echo "====== Table 4: Varying seq, 2 GPUs, causal ======"
printf "%-6s %-5s %-6s %-6s | %-10s %-6s | %-10s %-6s | %-10s %-6s | %-10s %-6s | %-10s\n" \
       "seq" "GPUs" "mask" "blocks" "RR(ms)" "imb" "BlkLPT" "imb" "RowLPT" "imb" "DP(ms)" "imb" "ideal"
echo "-----------------------------------------------------------------------------------------------------------"
$DEMO --seq  512 --gpus 2 --mask causal --alpha 0.169 --beta 0.00000694
$DEMO --seq 1024 --gpus 2 --mask causal --alpha 1.21  --beta 0.00000384
$DEMO --seq 2048 --gpus 2 --mask causal --alpha 1.21  --beta 0.00000384
$DEMO --seq 4096 --gpus 2 --mask causal --alpha 1.21  --beta 0.00000384

echo ""
echo "====== Table 5: Varying GPUs, seq=4096, causal ======"
printf "%-6s %-5s %-6s %-6s | %-10s %-6s | %-10s %-6s | %-10s %-6s | %-10s %-6s | %-10s\n" \
       "seq" "GPUs" "mask" "blocks" "RR(ms)" "imb" "BlkLPT" "imb" "RowLPT" "imb" "DP(ms)" "imb" "ideal"
echo "-----------------------------------------------------------------------------------------------------------"
$DEMO --seq 4096 --gpus 2 --mask causal --alpha 1.21 --beta 0.00000384
$DEMO --seq 4096 --gpus 3 --mask causal --alpha 1.21 --beta 0.00000384
$DEMO --seq 4096 --gpus 4 --mask causal --alpha 1.21 --beta 0.00000384
$DEMO --seq 4096 --gpus 8 --mask causal --alpha 1.21 --beta 0.00000384

echo ""
echo "====== Table 6: Causal vs Full, seq=4096, 2 GPUs ======"
printf "%-6s %-5s %-6s %-6s | %-10s %-6s | %-10s %-6s | %-10s %-6s | %-10s %-6s | %-10s\n" \
       "seq" "GPUs" "mask" "blocks" "RR(ms)" "imb" "BlkLPT" "imb" "RowLPT" "imb" "DP(ms)" "imb" "ideal"
echo "-----------------------------------------------------------------------------------------------------------"
$DEMO --seq 4096 --gpus 2 --mask full   --alpha 1.21  --beta 0.00000384
$DEMO --seq 4096 --gpus 2 --mask causal --alpha 1.21  --beta 0.00000384