wget -P ./data https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -P ./data https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
python scripts.py --input-path=./train-v2.0.json --output-path=./train.txt
python scripts.py --input-path=./dev-v2.0.json --output-path=./dev.txt
