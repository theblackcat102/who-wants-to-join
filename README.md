# Notes


1. Meetup v1

Original dataset from attention group recommendation

Group size 16.66
Total Group 13,390
Users 42,747

 13,390 2,705  5.22 4.95# who-wants-to-join

pip install git+https://github.com/phanein/deepwalk.git

deepwalk --workers 20  --input lj/com-lj.ungraph.txt --output lj/lj.rand.embeddings

deepwalk --workers 8  --input amazon/com-amazon.ungraph.txt --output amazon/amazon.rand.embeddings --walk-length 40 --window-size 10
