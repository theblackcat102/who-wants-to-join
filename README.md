# Prepare dataset


Amazon  : src/amazon.py
    
    1. bash download.sh

    2. cd data/amazon : unzip tgz files under amazon


Meetup NY/SF : src/meetup.py

    1. mkdir meetup_v2 under project root
    2. download meetup csv from here [https://www.kaggle.com/sirpunch/meetups-data-from-meetupcom](https://www.kaggle.com/sirpunch/meetups-data-from-meetupcom)
    3. Run ``` python -m src.meetup_trainer --dataset NY ```


 Aminer  : src/aminer.py

    1. mkdir aminer under project root
    2. download citation network v1 from here [https://www.aminer.cn/citation](https://www.aminer.cn/citation)
    3. unzip zip and rename the txt file as dblp.txt
    4. Run ``` python -m src.aminer_trainer ```


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

Data expansion using omdb api ( mainly extract more actors and movie plot text)

After expansion 7598/11615 actors only participate in one movie


Movie:
    1. language
    2. Director
    4. Writer
    5. country
    3. Production company
    4. Budget


gcn/amazon_hete_2020-01-28-11-12-59

    * All SAGEConv layers


gcn/amazon_hete_2020-01-28-11-16-52

    * GAT layers with last is GCN


gcn/amazon_hete_2020-01-28-12-14-34

    * with node classification

gcn/amazon_hete_2020-01-28-12-20-42
gcn/amazon_hete_2020-01-28-12-35-02

    * remove member prediction only on target node


https://www.kernix.com/article/community-detection-in-social-networks/

```
def calculFScore(i,j):
    i=[int(x) for x in i]
    j=[int(x) for x in j]
    inter=set(i).intersection(set(j))
    precision=len(inter)/float(len(j))
    recall=len(inter)/float(len(i))
    if recall==0 and precision==0:
        fscore=0
    else:
        fscore=2*(precision*recall)/(precision+recall)
    return fscore
```

# Baseline 

You need to create edge lists of each datasets first and pre-train using [graphvite](https://graphvite.io/docs/latest/introduction.html)

Pretrained [here](https://drive.google.com/drive/folders/1Pd4AdVtQxloQw1K17i1DqcMeckYj_YnS?usp=sharing)

## Deepwalk Avg Similarity

```
python -m dataset.aminer
sort aminer_train_edgelist.txt | uniq -u > aminer_dup_train_edgelist.txt
python -m graphvite_embeddings.test --edge-file aminer_dup_train_edgelist.txt --model DeepWalk 
```

Aminer

```
python -m baseline.similarity --dataset aminer --embeddings ./graphvite_embeddings/aminer_dup_train_edgelist.txt-64-DeepWalk.pkl --top-k 5
python -m baseline.similarity --dataset aminer --embeddings ./graphvite_embeddings/aminer_dup_train_edgelist.txt-64-DeepWalk.pkl --top-k 10
python -m baseline.similarity --dataset aminer --embeddings ./graphvite_embeddings/aminer_dup_train_edgelist.txt-64-DeepWalk.pkl --top-k 20
```


top 5 0.10709204089265839 0.16486902927580893 0.079301489470981

top 10 0.1197822546323775 0.25359527478171545 0.07840878518844621

top 20 0.12737309692167348 0.33564458140729325 0.07860052769105516


### SF Meetup
```
conda activate base
python -m graphvite_embeddings.test --edge-file 94101_dup_train_edgelist.txt --model DeepWalk 
conda deactivate
python -m baseline.similarity --dataset meetup --embeddings ./graphvite_embeddings/94101_dup_train_edgelist.txt-64-DeepWalk.pkl --city SF
```

top 5  0.001243265644426025 0.0007158196134574087 0.004724409448818898

top 10 0.001923967648840275 0.0012079455977093772 0.0047244094488188984

top 20 0.002813184263569264 0.0020029183415010187 0.0047244094488188984

## NY Meetup



```
python -m graphvite_embeddings.test --edge-file 94101_dup_train_edgelist.txt --model Deepwalk 
python -m baseline.similarity --dataset meetup --embeddings ./graphvite_embeddings/10001_dup_train_edgelist.txt-64-LINE.pkl --city NY --top-k 5
python -m baseline.similarity --dataset meetup --embeddings ./graphvite_embeddings/10001_dup_train_edgelist.txt-64-LINE.pkl --city NY --top-k 10
python -m baseline.similarity --dataset meetup --embeddings ./graphvite_embeddings/10001_dup_train_edgelist.txt-64-LINE.pkl --city NY --top-k 20
```

top 5  0.01787767020750293 0.017648179912330853 0.018113207547169816

top 10 0.01735544923908339 0.022927387078330475 0.013962264150943397

top 20 0.01471070344923796 0.02327044025157233 0.010754716981132076


## Deepwalk Avg Ranking

Only Aminer dataset for now

```
python -m baseline.deepwalk_clf
```


## Deepwalk + AGREE

```
CUDA_VISIBLE_DEVICES=1 python -m baseline.deepwalk_rank --embeddings graphvite_embeddings/aminer_deduplicate_train_edgelist.txt-64-DeepWalk.pkl --dataset aminer --top-k 5 --epochs 20 --neg-sample 5 --lr 0.000005
CUDA_VISIBLE_DEVICES=1 python -m baseline.deepwalk_rank --embeddings graphvite_embeddings/aminer_deduplicate_train_edgelist.txt-64-DeepWalk.pkl --dataset aminer --top-k 20 --epochs 50 --neg-sample 2 --lr 0.00005
CUDA_VISIBLE_DEVICES=1 python -m baseline.deepwalk_rank --embeddings graphvite_embeddings/aminer_deduplicate_train_edgelist.txt-64-DeepWalk.pkl --dataset aminer --top-k 10 --epochs 20 --neg-sample 5 --lr 0.000005
```


```
python -m baseline.deepwalk_agree --top-k 10
python -m baseline.deepwalk_agree --top-k 5
```