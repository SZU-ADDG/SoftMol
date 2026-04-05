# Search Strategy Evaluation Summary

Hit metrics use only QED/SA filtering + rv threshold (no SIM filtering).

## Hit ratio

| Method | parp1 | fa7 | 5ht1b | braf | jak2 |
|---|---|---|---|---|---|
| de novo | 0.04 | 0.02 | 0.26 | 0.01 | 0.07 |
| Greedy | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| Beam | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| MCTS | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Hit mean DS (within Hit)

| Method | parp1 | fa7 | 5ht1b | braf | jak2 |
|---|---|---|---|---|---|
| de novo | 10.60 | 8.83 | 9.59 | 10.60 | 9.45 |
| Greedy | 11.90 | 9.45 | 11.68 | 11.31 | 10.76 |
| Beam | 11.89 | 9.45 | 11.73 | 11.27 | 10.75 |
| MCTS | 12.59 | 9.89 | 12.34 | 11.92 | 11.28 |

## Top 5% DS (within Hit)

| Method | parp1 | fa7 | 5ht1b | braf | jak2 |
|---|---|---|---|---|---|
| de novo | 11.30 | 8.90 | 11.30 | 10.60 | 10.00 |
| Greedy | 12.90 | 10.18 | 12.76 | 12.14 | 11.50 |
| Beam | 12.86 | 10.20 | 12.87 | 12.19 | 11.57 |
| MCTS | 14.01 | 10.89 | 13.54 | 13.34 | 12.56 |

