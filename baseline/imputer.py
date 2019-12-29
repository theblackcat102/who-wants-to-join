from scipy.sparse.linalg import svds
from copy import deepcopy



def load_data(dataset='amazon', split_ratio=0.8, sample_ratio=0.5, 
        order_shuffle=True,
        train=True, query='group', pred_size=100, max_size=5000, min_size=10, min_freq=4):
    filename = '{}_freq_{}_{}-{}_{}_cache.pkl'.format(self.query, min_freq, max_size, min_size, dataset )
    with open(cache_path, 'rb') as f:
        ( group2user, member_map, member_frequency, keys ) = pickle.load(f)


