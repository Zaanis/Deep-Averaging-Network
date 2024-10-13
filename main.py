# models.py

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import CountVectorizer
from sentiment_data import read_sentiment_examples, read_word_embeddings
from torch.utils.data import Dataset, DataLoader
import time
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from BOWmodels import SentimentDatasetBOW, NN2BOW, NN3BOW
from DANmodels import DanData, DAN, collate_fn, DANSUB, BPE
from utils import Indexer
from tokenizers import ByteLevelBPETokenizer
from collections import Counter, defaultdict
from torch.utils.data import DataLoader, TensorDataset





# Training function
def train_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = train_loss / num_batches
    accuracy = correct / size
    return accuracy, average_train_loss


# Evaluation function
def eval_epoch(data_loader, model, loss_fn, optimizer):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    eval_loss = 0
    correct = 0
    for batch, (X, y) in enumerate(data_loader):
        X = X.float()

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        eval_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    average_eval_loss = eval_loss / num_batches
    accuracy = correct / size
    return accuracy, average_eval_loss


# Experiment function to run training and evaluation for multiple epochs
def experiment(model, train_loader, test_loader):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    all_train_accuracy = []
    all_test_accuracy = []
    for epoch in range(100):
        train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
        all_train_accuracy.append(train_accuracy)

        test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn, optimizer)
        all_test_accuracy.append(test_accuracy)

        if epoch % 10 == 9:
            print(f'Epoch #{epoch + 1}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')
    
    return all_train_accuracy, all_test_accuracy


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run model training based on specified model type')
    parser.add_argument('--model', type=str, required=True, help='Model type to train (e.g., DANGLOVE)')
    parser.add_argument('--file', type=int, default = 300, help='Path to the training file')
    parser.add_argument('--hidden_size', type=int, default=128, help='Size of the hidden layer (default: 128)')
    parser.add_argument('--vocab_size', type=int, default=1000, help='Size of the subdan vocab (default: 1000)')
    parser.add_argument('--num_merges', type=int, default=10, help='Number of merges of the most frequent pair of tokens (default: 10)')

    # Parse the command-line arguments
    args = parser.parse_args()




    # Check if the model type is "BOW"
    if args.model == "BOW":
        # Train and evaluate NN2
            # Load dataset
        start_time = time.time()

        train_data = SentimentDatasetBOW("data/train.txt")
        dev_data = SentimentDatasetBOW("data/dev.txt")
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        test_loader = DataLoader(dev_data, batch_size=16, shuffle=False)

        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Data loaded in : {elapsed_time} seconds")
        start_time = time.time()
        print('\n2 layers:')
        nn2_train_accuracy, nn2_test_accuracy = experiment(NN2BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Train and evaluate NN3
        print('\n3 layers:')
        nn3_train_accuracy, nn3_test_accuracy = experiment(NN3BOW(input_size=512, hidden_size=100), train_loader, test_loader)

        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_train_accuracy, label='2 layers')
        plt.plot(nn3_train_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for 2, 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(nn2_test_accuracy, label='2 layers')
        plt.plot(nn3_test_accuracy, label='3 layers')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for 2 and 3 Layer Networks')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")

        # plt.show()

    elif args.model == "DANGLOVE":
        def train_epoch(data_loader, model, loss_fn, optimizer):
            size = len(data_loader.dataset)
            num_batches = len(data_loader)
            model.train()
            train_loss, correct = 0, 0
            for batch, (X, y) in enumerate(data_loader):
                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)
                train_loss += loss.item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            average_train_loss = train_loss / num_batches
            accuracy = correct / size
            return accuracy, average_train_loss

        # Evaluation function
        def eval_epoch(data_loader, model, loss_fn):
            size = len(data_loader.dataset)
            num_batches = len(data_loader)
            model.eval()
            eval_loss = 0
            correct = 0
            with torch.no_grad():
                for batch, (X, y) in enumerate(data_loader):
                    # Compute prediction error
                    pred = model(X)
                    loss = loss_fn(pred, y)
                    eval_loss += loss.item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            average_eval_loss = eval_loss / num_batches
            accuracy = correct / size
            return accuracy, average_eval_loss

        # Experiment function to run training and evaluation for multiple epochs
        def experiment(model, train_loader, test_loader):
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            all_train_accuracy = []
            all_test_accuracy = []
            epochs = 50

            for epoch in range(epochs):
                train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
                all_train_accuracy.append(train_accuracy)

                test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn)
                all_test_accuracy.append(test_accuracy)

                print(f'Epoch {epoch + 1}/{epochs}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')

            # Plotting train and test accuracy over epochs
            return all_train_accuracy, all_test_accuracy
        # getting which GLOVE file to use
        if args.file == 50:
            g_file = "data/glove.6B.50d-relativized.txt"
        elif args.file == 300:
            g_file = "data/glove.6B.300d-relativized.txt"
            
        word_embeddings = read_word_embeddings(g_file)
        embedding_layer = word_embeddings.get_initialized_embedding_layer(frozen=True)
        embedding_dim = word_embeddings.get_embedding_length()
        
        hidden_dim = args.hidden_size
        output_dim = 2  
        train_examples = read_sentiment_examples("data/train.txt")  
        dev_examples = read_sentiment_examples("data/dev.txt")
        train_dataset = DanData(train_examples)
        dev_dataset = DanData(dev_examples)

        batch_size = 32
        train_loader_glove = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, word_embeddings.word_indexer))
        dev_loader_glove = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, word_embeddings.word_indexer))
        
        model = DAN(embedding_layer, embedding_dim, hidden_dim)

        start_time = time.time()
        train_acc, test_acc = experiment(model, train_loader_glove, dev_loader_glove)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\nModel trained in {elapsed_time} seconds")
        
            # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(train_acc, label='training')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'DANGLOVE_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(test_acc, label='Dev Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'DANGLOVE_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")
    
    elif args.model == "DANRANDOM":
        def train_epoch(data_loader, model, loss_fn, optimizer):
            size = len(data_loader.dataset)
            num_batches = len(data_loader)
            model.train()
            train_loss, correct = 0, 0
            for batch, (X, y) in enumerate(data_loader):
                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)
                train_loss += loss.item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            average_train_loss = train_loss / num_batches
            accuracy = correct / size
            return accuracy, average_train_loss

        # Evaluation function
        def eval_epoch(data_loader, model, loss_fn):
            size = len(data_loader.dataset)
            num_batches = len(data_loader)
            model.eval()
            eval_loss = 0
            correct = 0
            with torch.no_grad():
                for batch, (X, y) in enumerate(data_loader):
                    # Compute prediction error
                    pred = model(X)
                    loss = loss_fn(pred, y)
                    eval_loss += loss.item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            average_eval_loss = eval_loss / num_batches
            accuracy = correct / size
            return accuracy, average_eval_loss

        # Experiment function to run training and evaluation for multiple epochs
        def experiment(model, train_loader, test_loader):
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            all_train_accuracy = []
            all_test_accuracy = []
            epochs = 50

            for epoch in range(epochs):
                train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
                all_train_accuracy.append(train_accuracy)

                test_accuracy, test_loss = eval_epoch(test_loader, model, loss_fn)
                all_test_accuracy.append(test_accuracy)

                print(f'Epoch {epoch + 1}/{epochs}: train accuracy {train_accuracy:.3f}, dev accuracy {test_accuracy:.3f}')

            # Plotting train and test accuracy over epochs
            return all_train_accuracy, all_test_accuracy
       
        #new indexer object to map words to indices in the helper function
        random_word_indexer = Indexer()
        random_word_indexer.add_and_get_index("PAD")
        random_word_indexer.add_and_get_index("UNK")
        for example in read_sentiment_examples("data/train.txt"):
            for word in example.words:
                random_word_indexer.add_and_get_index(word)

        vocab_size = len(random_word_indexer)
        
        train_examples = read_sentiment_examples("data/train.txt")  
        dev_examples = read_sentiment_examples("data/dev.txt")
        train_dataset = DanData(train_examples)
        dev_dataset = DanData(dev_examples)
        batch_size = 32
        train_loader_random = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_fn(x, random_word_indexer))
        dev_loader_random = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_fn(x, random_word_indexer))
        word_embeddings = read_word_embeddings("data/glove.6B.300d-relativized.txt")
        embedding_dim = word_embeddings.get_embedding_length()        
        vocab_size = len(word_embeddings.word_indexer)
        embedding_dim = 300
        hidden_dim = 256
        random_embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        model_random = DAN(random_embedding_layer, embedding_dim, hidden_dim)
        optimizer = torch.optim.Adam(model_random.parameters(), lr=0.001)
        
        start_time = time.time()
        train_acc, test_acc = experiment(model_random, train_loader_random, dev_loader_random)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\nModel trained in {elapsed_time} seconds")
        
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(train_acc, label='training')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'DANRANDOM_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(test_acc, label='Dev Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'DANRANDOM_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")   
        
    elif args.model == "DANSUB":
        def train_epoch(data_loader, model, loss_fn, optimizer):
            size = len(data_loader.dataset)
            num_batches = len(data_loader)
            model.train()
            train_loss, correct = 0, 0
            for X, y in data_loader:
                X = X.long()

                # Compute prediction error
                pred = model(X)
                loss = loss_fn(pred, y)
                train_loss += loss.item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            average_train_loss = train_loss / num_batches
            accuracy = correct / size
            return accuracy, average_train_loss

        def eval_epoch(data_loader, model, loss_fn):
            size = len(data_loader.dataset)
            num_batches = len(data_loader)
            model.eval()
            eval_loss, correct = 0, 0
            with torch.no_grad():
                for X, y in data_loader:
                    X = X.long()

                    # Compute prediction error
                    pred = model(X)
                    loss = loss_fn(pred, y)
                    eval_loss += loss.item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            average_eval_loss = eval_loss / num_batches
            accuracy = correct / size
            return accuracy, average_eval_loss

        # Experiment function to run training and evaluation
        def experiment(model, train_loader, eval_loader):
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            all_train_accuracy = []
            all_eval_accuracy = []
            for epoch in range(50):
                train_accuracy, train_loss = train_epoch(train_loader, model, loss_fn, optimizer)
                all_train_accuracy.append(train_accuracy)

                eval_accuracy, eval_loss = eval_epoch(eval_loader, model, loss_fn)
                all_eval_accuracy.append(eval_accuracy)

                print(f'Epoch #{epoch + 1}: train accuracy: {train_accuracy:.3f}, dev accuracy: {eval_accuracy:.3f}')
            return all_train_accuracy, all_eval_accuracy

        train_file_path = 'data/train.txt'
        with open(train_file_path, 'r') as f:
            train_data = f.readlines()

        # Preprocess training data to build initial vocabulary
        X_train = [line.strip().split('\t')[1] for line in train_data]
        y_train = [int(line.strip().split('\t')[0]) for line in train_data]
        vocab = Counter()
        for sentence in X_train:
            tokens = sentence.split()  
            vocab.update(tokens)

        # set vocab size
        vocab = dict(vocab.most_common(args.vocab_size))

        num_merges = args.num_merges
        bpe = BPE(vocab, num_merges)
        bpe.learn_bpe()

        # Encode training data
        encoded_X_train = [bpe.encode(' '.join(sentence.split())) for sentence in X_train]
        X_train_tensor = [torch.tensor(sentence, dtype=torch.long) for sentence in encoded_X_train]
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        # Load evaluation data from file
        eval_file_path = 'data/dev.txt'
        with open(eval_file_path, 'r') as f:
            eval_data = f.readlines()

        # Preprocess and encode evaluation data
        X_eval = [line.strip().split('\t')[1] for line in eval_data]
        y_eval = [int(line.strip().split('\t')[0]) for line in eval_data]
        encoded_X_eval = [bpe.encode(' '.join(sentence.split())) for sentence in X_eval]
        X_eval_tensor = [torch.tensor(sentence, dtype=torch.long) for sentence in encoded_X_eval]
        y_eval_tensor = torch.tensor(y_eval, dtype=torch.long)

        # Create DataLoader for training and evaluation
        train_dataset = TensorDataset(torch.nn.utils.rnn.pad_sequence(X_train_tensor, batch_first=True), y_train_tensor)
        eval_dataset = TensorDataset(torch.nn.utils.rnn.pad_sequence(X_eval_tensor, batch_first=True), y_eval_tensor)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=32)

        # Model parameters
        vocab_size = len(bpe.word2index)
        embedding_dim = 300
        hidden_dim = args.hidden_size
        output_dim = 2

        # Initialize model, loss function, and optimizer
        model = DANSUB(vocab_size, embedding_dim, hidden_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        
        start_time = time.time()
        train_acc, test_acc = experiment(model, train_loader, eval_loader)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"\n\nModel trained in {elapsed_time} seconds")
        
        # Plot the training accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(train_acc, label='training')
        plt.xlabel('Epochs')
        plt.ylabel('Training Accuracy')
        plt.title('Training Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the training accuracy figure
        training_accuracy_file = 'DANSUB_train_accuracy.png'
        plt.savefig(training_accuracy_file)
        print(f"\n\nTraining accuracy plot saved as {training_accuracy_file}")

        # Plot the testing accuracy
        plt.figure(figsize=(8, 6))
        plt.plot(test_acc, label='Dev Acc')
        plt.xlabel('Epochs')
        plt.ylabel('Dev Accuracy')
        plt.title('Dev Accuracy for DAN')
        plt.legend()
        plt.grid()

        # Save the testing accuracy figure
        testing_accuracy_file = 'DANSUB_dev_accuracy.png'
        plt.savefig(testing_accuracy_file)
        print(f"Dev accuracy plot saved as {testing_accuracy_file}\n\n")   
if __name__ == "__main__":
    main()
