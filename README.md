# Notes


1. Meetup v1

Original dataset from attention group recommendation

Group size 16.66
Total Group 13,390
Users 42,747

 13,390 2,705  5.22 4.95# who-wants-to-join

# Setup

virtualenv -p python3.6 env

pip install -r requirements.txt

pip install git+https://github.com/phanein/deepwalk.git

## Download dataset

bash download.sh

## Run Deepwalk pre-training

deepwalk --workers 20  --input lj/com-lj.ungraph.txt --output lj/lj.rand.embeddings

deepwalk --workers 8  --input amazon/com-amazon.ungraph.txt --output amazon/amazon.rand.embeddings --walk-length 40 --window-size 10 

deepwalk --workers 8  --input amazon/com-amazon.ungraph.txt --output amazon/amazon.deep.32.embeddings --walk-length 40 --window-size 10 --representation-size 32

deepwalk --workers 8  --input lj/com-lj.ungraph.txt --output lj/lj.rand.embeddings --walk-length 40 --window-size 10


# Train

### Siamese Network

python -m baseline.train_sia --dataset dblp --task socnet --neg-ratio 5


### Seq2Seq

python -m baseline.train --dataset amazon --task socnet

### Deepset

python -m baseline.deepset3 --dataset dblp --task socnet

### Nearest Neighbour

python -m baseline.cluster


# Notes

Amazon

1. Seq2Seq

Pretrained graph embedding: Deepwalk do cluster similar groups under PCA

    * using pretrain on deepset, set transformer doesnt work ( same issue as before )

    * 64 dim 

Sort by occurance frequency: 

    * No diff on deepset, set transformer ( both model output, input doesn't really rely on sequence order )

    * Lower validation loss non sorted seq2seq version

L2 regularization : No difference test on 1e-5, 1e-3

    * higher validation loss

    * 1e-3 all metrics is nan


Presample and train later: sample 20k from training dataset as input and output sequences (similar to how deepwalk implements). 
    
    * lower training loss on seq2seq




