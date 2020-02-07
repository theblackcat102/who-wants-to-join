import json
from tqdm import tqdm
from collections import defaultdict
from torch_geometric.data import DataLoader
from src.amazon import AmazonCommunity
from src.meetup import Meetup, locations_id
from src.aminer import Aminer


dataset_classes = [AmazonCommunity, Meetup, Aminer]
default_kwargs = dict(cutoff=2, min_size=5, max_size=100)
meetup_city = ['SF', 'NY']
datasets = []
for dataset_class in dataset_classes:
    if dataset_class == Meetup:
        for city in meetup_city:
            default_kwargs['city_id'] = locations_id[city]
            datasets.append(Meetup(**default_kwargs))
            del default_kwargs['city_id']
    else:
        datasets.append(dataset_class(**default_kwargs))

all_stats = {}
for dataset in datasets:
    # setup
    if isinstance(dataset, Meetup):
        dataset_name = type(dataset).__name__ + "_{}".format(dataset.city_id)
    else:
        dataset_name = type(dataset).__name__
    print(dataset_name + " start!")
    all_stats[dataset_name] = defaultdict(int)
    stats = all_stats[dataset_name]
    stats["dataset"] = dataset_name
    stats["num_nodes"] = 0
    stats["num_edges"] = 0
    stats["num_groups"] = 0
    stats["num_member_nodes"] = 0
    # stats["total_group_size"] = 0
    loader = DataLoader(dataset, batch_size=16, num_workers=4)
    for data in tqdm(loader, dynamic_ncols=True):
        stats["num_nodes"] += len(data.x)
        stats["num_edges"] += len(data.edge_index[1])
        stats["num_groups"] += data.num_graphs
        stats["num_member_nodes"] += len(data.x[data.x[:, -1] == 0, 0])
    stats["avg_degree"] = stats["num_edges"] / stats["num_nodes"]
    stats["avg_group_size"] = stats["num_member_nodes"] / stats["num_groups"]
    print(dataset_name+" finish!")


# dumps to file
all_stats_json = json.dumps(all_stats, indent=4)
print(all_stats_json)
with open("stats.txt", "w") as file:
    file.write(all_stats_json)
