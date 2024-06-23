import torch
from torchtext.legacy import data
from torchtext.legacy import datasets
from konlpy.tag import Komoran
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time


class FastText(nn.Module):

    def __init__(self, vocab_size, embedding_dim, output_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, text):
        # text = [sent_len, batch_size]
        # print_shape('text', text)

        embedded = self.embedding(text)
        # print_shape('embedded', embedded)
        # embedded = [sent_len, batch_size, embedding_dim]

        # CNN은 [batch_size, sent_len, embedding_dim] 를 입력으로 받음
        # 따라서 permute 취해줘야 함
        embedded = embedded.permute(1, 0, 2)
        # print_shape('embedded', embedded)
        # embedded = [batch_size, sent_len, embedding_dim]

        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)
        # print_shape('pooled', pooled)
        # pooled = [batch_size, embedding_dim]

        res = self.fc(pooled)
        # print_shape('res', res)
        # res = [batch_size, output_dim]
        return res

def generate_bigrams(x):
    n_grams = set(zip(*[x[i:] for i in range(2)]))
    for n_gram in n_grams:
        x.append(' '.join(n_gram))
    return x

def main():

    x = ['너', '임마', '밥은', '먹고', '다니냐']
    n_grams = set(zip(*[x[i:] for i in range(2)]))

    generate_bigrams(['너', '임마', '밥은', '먹고', '다니냐'])


    komoran = Komoran()

    SEED = 1234

    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    TEXT = data.Field(tokenize=komoran.morphs, preprocessing=generate_bigrams)
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

    import random

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


    torch.save(TEXT.vocab, 'TEXTVocab.pt')
    torch.save(TEXT.pad_token, 'TEXTPad_token.pt')
    torch.save(LABEL, 'LABEL.pt')

    def print_shape(name, data):
        print(f'{name} has shape {data.shape}')

    txt = torch.rand(2, 5, 10)
    txt.shape, F.avg_pool2d(txt, (5, 1)).shape
    # (5 x 1) 크기의 필터를 옮겨가며 평균을 구한다.

    txt = torch.tensor(
        [[[1, 2, 3, 4], [4, 5, 6, 7]]], dtype=torch.float
    )
    print(txt.shape, "\n", txt)

    F.avg_pool2d(txt, (2, 1)).shape, F.avg_pool2d(txt, (2, 1))
    # (2 x 1) 필터로 평균을 취함

    F.avg_pool2d(txt, (2, 2)).shape, F.avg_pool2d(txt, (2, 2))
    # (2 x 2) 필터로 평균을 취함



    INPUT_DIM = len(TEXT.vocab)
    EMBEDDING_DIM = 300
    OUTPUT_DIM = 1
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

    model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'모델의 파라미터 수는 {count_parameters(model):,} 개 입니다.')

    pretrained_weight = TEXT.vocab.vectors
    model.embedding.weight.data.copy_(pretrained_weight)

    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

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
            torch.save(model.state_dict(), 'classification-model.pt')

        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')


def predict_sentiment(sentence):
    komoran = Komoran()

    TEXTVocab = torch.load('TEXTVocab.pt')
    TEXTPad_token = torch.load('TEXTPad_token.pt')

    INPUT_DIM = len(TEXTVocab)
    EMBEDDING_DIM = 300
    OUTPUT_DIM = 1
    PAD_IDX = TEXTVocab.stoi[TEXTPad_token]

    model = FastText(INPUT_DIM, EMBEDDING_DIM, OUTPUT_DIM, PAD_IDX)
    model.load_state_dict(torch.load('classification-model.pt'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    tokenized = generate_bigrams([tok for tok in komoran.morphs(sentence)])
    indexed = [TEXTVocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1) # 배치
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()


if __name__ == '__main__':
    print(predict_sentiment('이 영화 애매모호하네'))
    #main()
