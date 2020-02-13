import torch
import os, glob
from tqdm import tqdm
from multiprocessing import Pool


def convert(filename, file_directory, output_directory):
    basename = os.path.basename(filename)

    output_filename = basename.replace('.pt', '.txt')
    data = torch.load(filename)
    print(data)
    
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

def convert_data2txt(file_directory, output_directory):
    # pred_count, N = 0, 0
    # pool = Pool()
    os.makedirs(output_directory, exist_ok=True)

    for filename in tqdm(glob.glob(file_directory), dynamic_ncols=True):
        convert(filename, file_directory, output_directory)
    #     pool.apply_async(convert, args=(filename, file_directory, output_directory))

    # pool.close()
    # pool.join()

    # print(pred_count/N)

if __name__ == "__main__":
    # convert_data2txt('processed/dblp_v2/processed/dblp_2_*.pt', 'dblp_txt')
    # convert_data2txt('processed/amazon_hete/processed/amazon_2_0.8_5_hete_1*.pt', 'amazon_txt')
    # convert_data2txt('processed/10001/processed/*.pt', 'NY_txt')
    convert_data2txt('processed/94101/processed/*.pt', 'SF2_txt')
