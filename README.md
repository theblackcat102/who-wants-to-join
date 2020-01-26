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

deepwalk --workers 8  --input youtube/com-youtube.ungraph.txt --output youtube/youtube.rand.embeddings --walk-length 20 --window-size 10


python baseline.py --dataset amazon
python baseline.py --dataset dblp
python baseline.py --dataset youtube

# Train

### Siamese Network

python -m baseline.train_sia --dataset dblp --task socnet --neg-ratio 5

### Seq2Seq

python -m baseline.train --dataset amazon --task socnet

### Deepset

python -m baseline.deepset3 --dataset dblp --task socnet

### Nearest Neighbour

python -m baseline.cluster

### GCN

python -m src.trainer


# Tensorboard

tensorboard --logdir=logs


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



## Code review


There maybe some issue with my code?

Bug may lies in : Trainer section /  Dataset / Evaluation method

    * model design shouldn't have issue (just copy from repo)

    * pytorch lightning may not work as intented

    * training loss did lower 

[x] rewrite the trainer code -> result is same ( not pytorch-lightning issue )

[ ] check dataloader code ( pair review with james )

[ ] check F1 evaluation method ( pair review with james ) 



Facebook link:

https://monova.org/165bd18d9c34582509fd252ac45e930c9ebc9e8f

https://archive.org/details/oxford-2005-facebook-matrix

100 group as university

Node attribute:

1. Student faction
2. Gender
3. Major
4. Second Major
5. Dorm
6. Year
7. High school

Not sure what is this

http://law.di.unimi.it/webdata/fb-current/


Flickr:

1. http://socialcomputing.asu.edu/datasets/Flickr


Number of users : 80,513
Number of friendship pairs: 5,899,882
Number of groups: 195

2. http://socialnetworks.mpi-sws.org/data-imc2007.html


Github

Task : predict which member to add to project

follow, star, commit


https://www.gharchive.org/


~~MovieLens:~~

Movie:
    1. language
    2. Director
    4. Writer
    5. country
    3. Production company
    4. Budget


