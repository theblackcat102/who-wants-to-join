mkdir data
mkdir data/amazon
mkdir data/lj
mkdir data/friendster
mkdir data/youtube
mkdir data/orkut
mkdir data/dblp
wget https://snap.stanford.edu/data/bigdata/amazon/amazon-meta.txt.gz -P data/amazon
wget https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz -P data/amazon
wget https://snap.stanford.edu/data/bigdata/communities/com-amazon.all.dedup.cmty.txt.gz -P data/amazon
wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.all.cmty.txt.gz -P data/youtube
wget https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz -P data/youtube
wget https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz -P data/dblp
wget https://snap.stanford.edu/data/bigdata/communities/com-dblp.all.cmty.txt.gz -P data/dblp
