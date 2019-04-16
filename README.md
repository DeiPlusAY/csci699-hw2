## Usage
For running on CPU, use --gpu -1

Glove CNN:

python train.py --embedding "your glove txt file"


For BERT version, first open an bert-as-service instance. Then use the following command:

Bert Pooled:

python train.py --bert pooled

Bert Contextual:

python train.py --bert contextual
