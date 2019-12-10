'''

    Convert dataset to AGREE format

'''
import pickle
from .dataloader import extract_relation, create_inv_map
import os

def create_group_member(datapath='./meetup_v1'):
    user2group = extract_relation(os.path.join(datapath, 'user_groups.txt'))
    group2user = create_inv_map(user2group)
    with open('Attentive-Group-Recommendation/data/Meetup/groupMember.txt', 'w') as f:
        for key, value in group2user.items():
            if len(value) > 0:
                value_ = [ str(v) for v in value]
                f.write('{}\t{}\n'.format(str(key), ','.join(value_)))


def create_full_data(train=True):
    with open(os.path.join('.cache', 'meetup_v1_data_cache.pkl'), 'rb') as f:
        (data, group2tag, user2group, event2user, user2tag, group2user) = pickle.load(f)
    user2group = extract_relation(os.path.join(datapath, 'user_groups.txt'))
    group2user = create_inv_map(user2group)

    with open('Attentive-Group-Recommendation/data/Meetup/groupMember.txt', 'w') as f:
        for key, value in group2user.items():
            if len(value) > 0:
                value_ = [ str(v) for v in value]
                f.write('{}\t{}\n'.format(str(key), ','.join(value_)))
    
    data.sort_values(by='date' )
    
    if train:
        data = data[:int(len(data)*split_ratio)]
    else:
        data = data[int(len(data)*split_ratio):]



if __name__ == "__main__":
    create_group_member()