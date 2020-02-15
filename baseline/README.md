Rank based CF GNN baseline



1. Pretrain using context prediction from [STRATEGIES FOR PRE-TRAINING GRAPH NEURAL NETWORKS](https://github.com/snap-stanford/pretrain-gnns)

```
python -m baseline.pretrain_context
```

2. Train rank model
```
python -m baseline.rank [meetup/aminer/]
```