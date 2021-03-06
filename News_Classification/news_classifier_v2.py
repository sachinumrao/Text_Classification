# import dependencies
import os
import time
import re

import torch
import torchtext
from torchtext.datasets import text_classification
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer


# data params
NGRAMS = 2
BATCH_SIZE = 16

# load dataset
#os.mkdir('./.data')
train_dataset, test_dataset = text_classification.DATASETS['AG_NEWS'](
    root='./.data', ngrams=NGRAMS, vocab=None)

# check for cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# collate function used by dataloader
def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    
    text = torch.cat(text)
    return text, label

# Model class
class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()

        # gru unit params
        gru_hidden = 128
        gru_layers = 2

        # fully connected lyers params
        fc1_size = 256
        fc2_size = 64

        #self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.encoder = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, gru_hidden, gru_layers, 
                          batch_first=True, dropout=0.05, 
                          bidirectional=True)

        self.relu = nn.ReLU()

        self.batch_norm1 = nn.BatchNorm1d(fc1_size)

        # fully connected layers
        self.fc1 = nn.Linear(gru_hidden*2, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, num_class)

        # weight initialization
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.fc1.weight.data.uniform_(-initrange, initrange)
        self.fc1.bias.data.zero_()

        self.fc2.weight.data.uniform_(-initrange, initrange)
        self.fc2.bias.data.zero_()

        self.fc3.weight.data.uniform_(-initrange, initrange)
        self.fc3.bias.data.zero_()

    def forward(self, text, offsets):
        out, hidden = self.gru(self.encoder(text))
        out = out[:,-1,:]
        out = self.relu(self.batch_norm1(self.fc1(out)))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


# utility function to run training for one epoch
def train_func(sub_train_):

    # Train the model
    train_loss = 0
    train_acc = 0
    data = DataLoader(sub_train_, batch_size=BATCH_SIZE, shuffle=True,
                      collate_fn=generate_batch)
    for i, (text, offsets, cls) in enumerate(data):
        optimizer.zero_grad()
        text, labels = text.to(device), offsets.to(device), cls.to(device)
        output = model(text)
        loss = criterion(output, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += (output.argmax(1) == labels).sum().item()

    # Adjust the learning rate
    scheduler.step()

    return train_loss / len(sub_train_), train_acc / len(sub_train_)

# function to run model on tese(validation) data
def test(data_):
    loss = 0
    acc = 0
    data = DataLoader(data_, batch_size=BATCH_SIZE, collate_fn=generate_batch)
    for text, labels in data:
        text, labels = text.to(device), labels.to(device)
        with torch.no_grad():
            output = model(text)
            loss = criterion(output, labels)
            loss += loss.item()
            acc += (output.argmax(1) == labels).sum().item()

    return loss / len(data_), acc / len(data_)


# model params
VOCAB_SIZE = len(train_dataset.get_vocab())
EMBED_DIM = 32
NUN_CLASS = len(train_dataset.get_labels())

# training parmas
N_EPOCHS = 10
min_valid_loss = float('inf')

# create model object
model = TextSentiment(VOCAB_SIZE, EMBED_DIM, NUN_CLASS).to(device)

# create loss function and optimizer
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

train_len = int(len(train_dataset) * 0.95)
sub_train_, sub_valid_ = \
    random_split(train_dataset, [train_len, len(train_dataset) - train_len])

# training loop
for epoch in range(N_EPOCHS):

    start_time = time.time()
    train_loss, train_acc = train_func(sub_train_)
    valid_loss, valid_acc = test(sub_valid_)

    secs = int(time.time() - start_time)
    mins = secs / 60
    secs = secs % 60

    print('Epoch: %d' %(epoch + 1), " | time in %d minutes, %d seconds" %(mins, secs))
    print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
    print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')


print('Checking the results of test dataset...')
test_loss, test_acc = test(test_dataset)
print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')


# Test for sample news
# create labels for news classes
ag_news_label = {1 : "World",
                 2 : "Sports",
                 3 : "Business",
                 4 : "Sci/Tec"}

# predict funtion utility to preprocess news article and evaluate on model
def predict(text, model, vocab, ngrams):
    tokenizer = get_tokenizer("basic_english")
    with torch.no_grad():
        text = torch.tensor([vocab[token]
                            for token in ngrams_iterator(tokenizer(text), ngrams)])
        output = model(text)
        return output.argmax(1).item() + 1

# paste news item here
ex_text_str = '''
The EV market still lacking a fun Miata-size convertible and a rendering artist \
tried to imagine what it would look like if Tesla gave it a shot based on Model 3: \
Love it or Hate it?
There are all-electric roadsters on the market and more coming but they are \
mostly focused on the higher-end of the market.

Tesla has its next-gen Roadster coming and while the specs are impressive, \
we are talking about a $200,000+ car.

What we are talking about is a “Mazda Miata of electric cars.”

A light 2-seater with no more than a ~50 kWh battery pack, like the base \
Model 3, and it would still get over 200 miles of range thanks to its weight \
and form factor.

While a Mazda Miata price point would be hard to achieve, it could ad least \
be sold for under $50,000.

We haven’t seen many companies going for that market aside for Electra \
Meccanica with the Tofino, which checks a lot of those boxes, but the vehicle \
is still far from hitting the market.

Design editor and rendering artist Lem Bingley tried to imagine what it would \
look like if Tesla would try to make something in this segment based on the Model 3.

'''

# extract dictionary from train dataset
vocab = train_dataset.get_vocab()
model = model.to("cpu")

print("This is a %s news" %ag_news_label[predict(ex_text_str, model, vocab, 2)])
