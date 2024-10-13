To run the code, open the terminal and use command:

1. 
python main.py --model DANGLOVE

for the DAN model with GLOVE embedding layer (part 1a). 
Some other parameters that can be passed in are:
--file, with the options of 50 or 300. 50 uses the 50d glove embeddings while 300 uses the 300d glove embeddings, the default is 300.
--hidden_size, the size of the hidden layer in the DAN, the default is 128

2. 
python main.py --model DANRANDOM 

for the DAN model with randomly initilaized embedding layer (part 1b).
Some other parameters that can be passed in are:
--hidden_size, the size of the hidden layer in the DAN, the default is 128

3. 
python main.py --model DANSUB

for the DAN model with subword tokenization (part 2)
Some other parameters that can be passed in are:
--hidden_size, the size of the hidden layer in the DAN, the default is 128
--vocab_size,  the size of the subwords vocabulary, the default is 1000
--num_merges, the number of merges of the most frequent pair of tokens, the default is 10