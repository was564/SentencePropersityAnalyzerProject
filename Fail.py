import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchtext.legacy import data

def init_data():
    columns = ['id', 'text', 'label']
    train_data = pd.read_csv('data/ratings_train.txt', sep='\t', names=columns, skiprows=1).dropna()  # null데이터 삭제
    test_data = pd.read_csv('data/ratings_test.txt', sep='\t', names=columns, skiprows=1).dropna()

    train_data.to_csv('data/train_data.csv', index=False)
    test_data.to_csv('data/test_data.csv', index=False)

    from konlpy.tag import Komoran

    komoran = Komoran()

    print(komoran.morphs('이 밤 그날의 반딧불을 당신의 창 가까이 보낼게요'))

    import torch
    from torchtext.legacy import data

    TEXT = data.Field(tokenize=komoran.morphs)
    LABEL = data.LabelField(dtype=torch.float)

    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}
    # dictionary 형식은 {csv컬럼명 : (데이터 컬럼명, Field이름)}

    train_data, test_data = data.TabularDataset.splits(
        path='data',
        train='train_data.csv',
        test='test_data.csv',
        format='csv',
        fields=fields
    )

    print(vars(train_data[0]))

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters,
                                              kernel_size=(fs, embedding_dim))
                                    for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # print_shape('text', text)
        # text = [batch_size, sent_len]

        embedded = self.embedding(text)
        # print_shape('embedded', embedded)
        # embedded = [batch_size, sent_len, emb_dim]

        embedded = embedded.unsqueeze(1)
        # print_shape('embedded', embedded)
        # embedded = [batch_size, 1, sent_len, emb_dim]

        # print_shape('self.convs[0](embedded)', self.convs[0](embedded))
        # self.convs[0](embedded) = [batch_size, n_filters, sent_len-filter_sizes[n]+1, 1 ]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # print_shape('F.max_pool1d(conved[0], conved[0].shape[2])', F.max_pool1d(conved[0], conved[0].shape[2]))
        # F.max_pool1d(conved[0], conved[0].shape[2]) = [batch_size, n_filters, 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # print_shape('cat', cat)
        # cat = [batch_size, n_filters * len(filter_size)]

        res = self.fc(cat)
        # print_shape('res', res)
        # res = [batch_size, output_dim]

        return self.fc(cat)


def cnn():
    import torch
    from torchtext.legacy import data
    from torchtext.legacy import datasets
    import random
    import numpy as np
    from konlpy.tag import Komoran
    komoran = Komoran()

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    FILTER_SIZES = [3, 4, 5]

    def tokenizer(text):
        token = [t for t in komoran.morphs(text)]
        if len(token) < max(FILTER_SIZES):
            for i in range(0, max(FILTER_SIZES) - len(token)):
                token.append('<PAD>')
        return token

    TEXT = data.Field(tokenize=tokenizer, batch_first=True)
    LABEL = data.LabelField(dtype=torch.float)

    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}
    # dictionary 형식은 {csv컬럼명 : (데이터 컬럼명, Field이름)}

    import random
    train_data, test_data = data.TabularDataset.splits(
        path='data',
        train='train_data.csv',
        test='test_data.csv',
        format='csv',
        fields=fields,
    )
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    MAX_VOCAB_SIZE = 25000

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors='fasttext.simple.300d',
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)

    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)

    import torch.nn as nn
    import torch.nn.functional as F

    def print_shape(name, data):
        print(f'{name} has shape {data.shape}')

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    OUTPUT_DIM = 1
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    torch.save(TEXT.vocab, 'TEXTVocab.pt')
    torch.save(TEXT.pad_token, 'TEXTPad_token.pt')
    torch.save(LABEL, 'LABEL.pt')

    model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'모델의 파라미터 수는 {count_parameters(model):,} 개 입니다.')

    pretrained_weight = TEXT.vocab.vectors
    print(pretrained_weight.shape, model.embedding.weight.data.shape)

    model.embedding.weight.data.copy_(pretrained_weight)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    import torch.optim as optim

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    def binary_accuracy(preds, y):
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc

    def train(model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for batch in iterator:
            optimizer.zero_grad()
            predictions = model(batch.text).squeeze(1)  # output_dim = 1
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():
            for batch in iterator:
                predictions = model(batch.text).squeeze(1)
                loss = criterion(predictions, batch.label)
                acc = binary_accuracy(predictions, batch.label)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    import time

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    N_EPOCHS = 5
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut4-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def predict_sentiment(model, sentence):
    model.eval()
    tokenized = generate_bigrams([tok for tok in mecab.morphs(sentence)])
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1) # 배치
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

if __name__ == '__main__':
    #predict('진정한 쓰레기')
    cnn()







def rnn():
    import torch
    from torchtext.legacy import data

    SEED = 1234

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    from konlpy.tag import Komoran
    komoran = Komoran()

    TEXT = data.Field(tokenize=komoran.morphs)
    LABEL = data.LabelField(dtype=torch.float)

    fields = {'text': ('text', TEXT), 'label': ('label', LABEL)}
    # dictionary 형식은 {csv컬럼명 : (데이터 컬럼명, Field이름)}

    train_data, test_data = data.TabularDataset.splits(
        path='data',
        train='train_data.csv',
        test='test_data.csv',
        format='csv',
        fields=fields,
    )

    vars(train_data[0]), vars(train_data[1])

    import random
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))

    MAX_VOCAB_SIZE = 25000

    TEXT.build_vocab(train_data,
                     max_size=MAX_VOCAB_SIZE,
                     vectors='fasttext.simple.300d',
                     unk_init=torch.Tensor.normal_)

    LABEL.build_vocab(train_data)

    for i in range(len(train_data)):
        if len(train_data.examples[i].text) == 0: print(i)

    BATCH_SIZE = 64

    BATCH_SIZE = 64

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        # sort_within_batch = True,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)

    import torch
    import torch.nn as nn
    class RNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                     n_layers, bidirectional, dropout, pad_idx):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                               bidirectional=bidirectional, dropout=dropout)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, text, text_lengths):
            # text = [sent_len, batch_size]
            # print_shape('text',text)
            embedded = self.dropout(self.embedding(text))
            # embedded = [sent_len, batch_size, emb_dim]
            # print_shape('embedded', embedded)

            # pack sequence
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
            packed_output, (hidden, cell) = self.rnn(packed_embedded)
            # print_shape('packed_output', packed_output)
            # print_shape('hidden', hidden)
            # print_shape('cell', cell)

            # unpack sequence
            output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
            # print_shape('output', output)
            # print_shape('output_lengths', output_lengths)

            # output = [sent_len, batch_size, hi_dim * num_directions]
            # output over padding tokens are zero tensors
            # hidden = [num_layers * num_directions, batch_size, hid_dim]
            # cell = [num_layers * num_directions, batch_size, hid_dim]

            # concat the final forward and backward hidden layers
            # and apply dropout

            # print_shape('hidden[-2,:,:]', hidden[-2,:,:])
            # print_shape('hidden[-1,:,:]', hidden[-1,:,:])
            # cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
            # print_shape('cat', cat)

            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
            # print_shape('hidden', hidden)
            # hidden = [batch_size, hid_dim * num_directions]

            res = self.fc(hidden)
            # print_shape('res', res)
            return res

    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300  # fasttext dim과 동일하게 설정
    HIDDEN_DIM = 256
    OUTPUT_DIM = 1
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM,
           N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'이 모델은 {count_parameters(model):,} 개의 파라미터를 가지고 있다.')

    pretrained_embeddings = TEXT.vocab.vectors

    print(pretrained_embeddings.shape)

    model.embedding.weight.data.copy_(pretrained_embeddings)  # copy_ 메서드는 인수를 현재 모델의 웨이트에 복사함

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

    import torch.optim as optim

    optimizer = optim.Adam(model.parameters())

    criterion = nn.BCEWithLogitsLoss()

    model = model.to(device)
    criterion = criterion.to(device)

    def binary_accuracy(preds, y):
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()
        acc = correct.sum() / len(correct)
        return acc

    def train(model, iterator, optimizer, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.train()

        for batch in iterator:
            optimizer.zero_grad()
            text, text_lengths = batch.text
            predictions = model(text, text_lengths).squeeze(1)
            # print_shape('predictions',predictions)

            loss = criterion(predictions, batch.label)
            # print_shape('loss',loss)

            acc = binary_accuracy(predictions, batch.label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate(model, iterator, criterion):
        epoch_loss = 0
        epoch_acc = 0

        model.eval()

        with torch.no_grad():
            for batch in iterator:
                text, text_lengths = batch.text
                predictions = model(text, text_lengths).squeeze(1)

                loss = criterion(predictions, batch.label)  # .squeeze(0))
                acc = binary_accuracy(predictions, batch.label)  # .squeeze(0))

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    import time

    def epoch_time(start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    N_EPOCHS = 5
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'tut2-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')