import os
import argparse
import configparser

from sklearn import metrics
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils import CustomDataset, make_data, load_data, load_embedding
from cnn import CNNTextModel
from lstmattention import LSTMAttention

cfg = configparser.ConfigParser()
cfg.read("settings.ini")

TRAIN_FILE = cfg["file"]["train_file"].replace("/", os.path.sep)
TEST_FILE = cfg["file"]["test_file"].replace("/", os.path.sep)
PREDICT_FILE = cfg["file"]["predict_file"].replace("/", os.path.sep)
EMBEDDING_FILE = cfg["file"]["embedding_file"].replace("/", os.path.sep)
MODEL_DIR = cfg["file"]["model_dir"].replace("/", os.path.sep)
RESULT_DIR = cfg["file"]["result_dir"].replace("/", os.path.sep)

USE_CUDA = cfg["train"]["cuda"].lower() == "true"
BATCH_SIZE = int(cfg["train"]["batch_size"])
EPOCHS = int(cfg["train"]["epochs"])

MAX_LEN = int(cfg["process"]["max_sentence_len"])
MIN_COUNT = int(cfg["process"]["min_word_count"])

NUM_CLASSES = int(cfg["model"]["num_classes"])


def train():
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

    if not os.path.exists(MODEL_DIR):
        os.mkdir(MODEL_DIR)

    # Load data.
    print("Load data...")
    train_dataset = CustomDataset(file=TRAIN_FILE,
                                  max_len=MAX_LEN,
                                  min_count=MIN_COUNT,
                                  result_dir=RESULT_DIR)
    test_dataset = CustomDataset(file=TEST_FILE,
                                 max_len=MAX_LEN,
                                 min_count=MIN_COUNT,
                                 result_dir=RESULT_DIR)
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE)
    word_embedding = load_embedding(EMBEDDING_FILE, min_count=MIN_COUNT, result_dir=RESULT_DIR)
    print("Load data success.")

    # Build model.
    # model = CNNTextModel(word_embedding=word_embedding, num_classes=NUM_CLASSES)
    model = LSTMAttention(word_embedding=word_embedding, hidden_dim=300, num_classes=NUM_CLASSES)
    model.to(device)

    # Build optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train
    data_size = len(train_dataset)
    batch_num = data_size // BATCH_SIZE + 1
    for epoch in range(1, EPOCHS + 1):
        # Train model.
        model.train()
        batch = 1
        # if epoch < 1:
        #     model.word_embedding.weight.requires_grad = True
        for batch_xs, batch_ys in train_loader:
            batch_xs = batch_xs.to(device)  # (N, L)
            batch_ys = batch_ys.to(device)  # (N, )
            batch_out = model(batch_xs)  # (N, num_classes)
            loss = F.cross_entropy(batch_out, batch_ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                print("epoch {}, batch {}/{}, train loss {}".format(epoch, batch, batch_num, loss.item()))
            batch += 1
        checkpoint_path = os.path.join(MODEL_DIR, "model_epoch_{}.ckpt".format(epoch))
        torch.save(model, checkpoint_path)

        # Test model.
        model.eval()
        y_true = []
        y_pred = []
        for batch_xs, batch_ys in test_loader:
            batch_xs = batch_xs.to(device)  # (N, L)
            batch_ys = batch_ys.to(device)  # (N, )
            batch_out = model(batch_xs)  # (N, num_classes)
            batch_pred = batch_out.argmax(dim=-1)  # (N, )
            for i in batch_ys.numpy():
                y_true.append(i)
            for i in batch_pred.numpy():
                y_pred.append(i)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        print("epoch {}, test accuracy {}".format(epoch, accuracy))


def predict(epoch_idx):
    """Load model in `models` and predict."""
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")

    model = torch.load(os.path.join(MODEL_DIR, "model_epoch_{}.ckpt".format(epoch_idx)))
    model = model.to(device)
    model.eval()

    X, _ = load_data(PREDICT_FILE,
                     max_len=MAX_LEN,
                     min_count=MIN_COUNT,
                     result_dir=RESULT_DIR)
    X = torch.LongTensor(X).to(device)  # (N, L)
    out = model(X)  # (N, num_classes)
    pred = out.argmax(dim=-1)  # (N, )
    pred = pred.cpu().numpy()

    with open(os.path.join(RESULT_DIR, "predict.txt"), "w", encoding="utf-8") as fw:
        for label in pred:
            fw.write(str(label) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="preprocess", choices=["preprocess", "train", "predict"])
    parser.add_argument("--epoch-idx", type=int, default=1)

    args = parser.parse_args()

    if args.mode == "preprocess":
        make_data(train_file=TRAIN_FILE, result_dir=RESULT_DIR)
    elif args.mode == "train":
        train()
    else:
        predict(args.epoch_idx)