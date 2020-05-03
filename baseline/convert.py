import torch
import os, glob
from tqdm import tqdm
import numpy as np
import networkx as nx
from multiprocessing import Pool
from dataset.aminer import Aminer
import pickle
def convert(filename, file_directory, output_directory):
    basename = os.path.basename(filename)

    output_filename = basename.replace('.pt', '.txt')
    data = torch.load(filename)    
    with open(os.path.join(output_directory, output_filename), 'w') as f:
        if 'known' in data:
            known_nodes = [ '{}_{}'.format(0, int(n[0])) for n in data.x[ data.known == 1]]
        else:
            known_nodes = [ '{}_{}'.format(0, int(n[0])) for n in data.x[data.x[:, 1] == 1]]

        f.write(' '.join(known_nodes)+'\n')

        pred_nodes = [ '{}_{}'.format(0, int(n[0])) for n in data.x[data.y == 1] ]
        f.write(' '.join(pred_nodes)+'\n')
        # pred_count += len(pred_nodes)
        # N += 1
        for edge in data.edge_index.transpose(1, 0):
            src, dst = edge
            src_node_id, _, src_node_type = data.x[src]
            dst_node_id, _, dst_node_type = data.x[dst]
            f.write('{}_{} {}_{}\n'.format(src_node_type, src_node_id,  dst_node_type, dst_node_id))




def test_():
    filename = 'processed/10001/processed/meetups_10001_2_0.8_5_3_604_v2.pt'
    data = torch.load(filename)
    for edge in data.edge_index.transpose(1, 0):
        src, dst = edge
        src_node_id, _, src_node_type = data.x[src]
        dst_node_id, _, dst_node_type = data.x[dst]
        if src_node_type == 4 or dst_node_type == 4:
            print(edge)


def convert_data2txt(file_directory, output_directory):
    # pred_count, N = 0, 0
    pool = Pool()
    os.makedirs(output_directory, exist_ok=True)

    for filename in tqdm(glob.glob(file_directory), dynamic_ncols=True):
        # convert(filename, file_directory, output_directory)
        print(filename)
        pool.apply_async(convert, args=(filename, file_directory, output_directory))

    pool.close()
    pool.join()


def vincent_convert(filename, file_directory, output_directory):
    basename = os.path.basename(filename)

    output_filename = basename.replace('.pt', '.txt')
    data = torch.load(filename)    
    with open(os.path.join(output_directory, output_filename), 'w') as f:
        if 'known' in data:
            known_nodes = [ '{}_{}'.format(0, int(n[0])) for n in data.x[ data.known == 1]]
        else:
            known_nodes = [ '{}_{}'.format(0, int(n[0])) for n in data.x[data.x[:, 1] == 1]]

        f.write(' '.join(known_nodes)+'\n')

        pred_nodes = [ '{}_{}'.format(0, int(n[0])) for n in data.x[data.y == 1] ]
        f.write(' '.join(pred_nodes)+'\n')

def vincent_convert_data2txt():
    # pred_count, N = 0, 0
    node_id = {
        'a': 0,
        'p': 1,
        'c': 2,
    }
    # def convert_name2node(n):
    #     return str(node_id[n[0]]) + '_' + n[1:] 

    # with open('processed/dblp_vincent/init_social_network.pkl', 'rb') as f:
    #     H = pickle.load(f)

    # fh = open("init_social_network.edgelist",'w')
    # # p, a, c
    # for u,v in H.edges(data=False):
    #     fh.write('{} {}\n'.format(convert_name2node(u), convert_name2node(v)))

    pool = Pool()

    file_directory = 'processed/dblp_vincent/processed/dblp_2_*.pt'
    output_directory = 'dblp_vincent_test_no_social'
    os.makedirs(output_directory, exist_ok=True)
    for filename in tqdm(glob.glob(file_directory), dynamic_ncols=True):
        # convert(filename, file_directory, output_directory)
        print(filename)
        pool.apply_async(vincent_convert, args=(filename, file_directory, output_directory))

    pool.close()
    pool.join()



def sample_aminer_test():
    dataset = Aminer()
    data_size = len(dataset)
    train_split, val, test = int(data_size*0.7), int(data_size*0.1), int(data_size*0.2)

    test_indexes = np.array(list(range(data_size)), dtype=np.long)[train_split+val:]
    test_dataset = dataset[list(test_indexes)]
    # print(test_dataset.processed_file_idx[:100])
    file_directory = [ os.path.join(test_dataset.processed_dir, filename) for filename in test_dataset.processed_file_idx]
    output_directory = 'dblp_txt_test'
    pool = Pool()
    os.makedirs(output_directory, exist_ok=True)

    for filename in tqdm(file_directory, dynamic_ncols=True):
        # convert(filename, file_directory, output_directory)
        print(filename)
        pool.apply_async(convert, args=(filename, file_directory, output_directory))

    pool.close()
    pool.join()


    # print(pred_count/N)

if __name__ == "__main__":
    vincent_convert_data2txt()
    # test_()
    # sample_aminer_test()
    # convert_data2txt('processed/dblp_vincent/processed/dblp_2_*.pt', 'dblp_vincent')
    # convert_data2txt('processed/amazon_hete/processed/amazon_2_0.8_5_hete_1*.pt', 'amazon_txt')
    # convert_data2txt('processed/10001/processed/meetup*.pt', '/mnt/HDD/NY_txt')
    # convert_data2txt('processed/94101/processed/meetup*.pt', '/mnt/HDD/SF2_txt')
