import os
import re
from statistics import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    set_seed,
)
from torch.optim import AdamW
from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)


TRAIN_CONLLU = (
    r"TRAIN.conllu"
)
DEV_CONLLU = (
    r"DEV.conllu"
)
TEST_CONLLU = (
    r"TEST.conllu"
)

BASE_SAVE_DIR = (
    r"C:\Users\usman\PycharmProjects\nersaraiki"
    r"\ner_model"
)
os.makedirs(BASE_SAVE_DIR, exist_ok=True)


BERT_NAME = "distilbert/distilbert-base-multilingual-cased"


SEEDS = [42, 123, 456, 789, 2026]

BATCH_SIZE = 8
EPOCHS = 10
LR = 2e-5
MAX_LEN = 256
PATIENCE = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def get_ner(misc):
    for x in misc.split("|"):
        if x.startswith("NER="):
            return x.replace("NER=", "")
    return "O"


def read_conllu(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        text = f.read().strip()

    sents = []

    for block in re.split(r"\n\n+", text):
        sent = []

        for line in block.split("\n"):
            if not line or line.startswith("#"):
                continue

            cols = line.split("\t")

            if len(cols) != 10:
                continue

            if "-" in cols[0] or "." in cols[0]:
                continue

            sent.append((cols[1], get_ner(cols[9])))

        if sent:
            sents.append(sent)

    return sents


class Vocab:
    def __init__(self, labels):
        labels = sorted(set(labels))
        self.lab2id = {x: i for i, x in enumerate(labels)}
        self.id2lab = {i: x for x, i in self.lab2id.items()}


class NERDataset(Dataset):
    def __init__(self, sents, vocab):
        self.data = []

        for sent in sents:
            tokens = [x[0] for x in sent]
            labels = [vocab.lab2id[x[1]] for x in sent]
            self.data.append((tokens, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def collate(batch):
    tokens, labels = zip(*batch)

    enc = tokenizer(
        list(tokens),
        is_split_into_words=True,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    y = []

    for i in range(len(tokens)):
        word_ids = enc.word_ids(batch_index=i)
        prev = None
        row = []

        for wid in word_ids:
            if wid is None:
                row.append(-100)
            elif wid != prev:
                row.append(labels[i][wid])
            else:
                row.append(-100)

            prev = wid

        y.append(row)

    return enc, torch.tensor(y, dtype=torch.long)


class BertNER(nn.Module):
    def __init__(self, model_name, n_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(
            self.bert.config.hidden_size,
            n_labels,
        )

    def forward(self, enc, labels=None):
        out = self.bert(**enc)
        logits = self.fc(self.drop(out.last_hidden_state))

        loss = None

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return loss, logits


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    true_labels = []
    pred_labels = []

    for enc, y in tqdm(loader, desc="Eval"):
        enc = {k: v.to(device) for k, v in enc.items()}
        y = y.to(device)

        _, logits = model(enc)
        pred = logits.argmax(-1)

        y = y.cpu().numpy()
        pred = pred.cpu().numpy()

        for i in range(len(y)):
            sent_true = []
            sent_pred = []

            for j in range(len(y[i])):
                if y[i][j] != -100:
                    sent_true.append(
                        vocab.id2lab[int(y[i][j])]
                    )
                    sent_pred.append(
                        vocab.id2lab[int(pred[i][j])]
                    )

            true_labels.append(sent_true)
            pred_labels.append(sent_pred)

    # Same default seqeval behavior as your original code.
    p = precision_score(true_labels, pred_labels)
    r = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels)

    return p, r, f1, report


def write_summary(results):
    lines = [
        f"Model: {BERT_NAME}",
        f"Seeds: {SEEDS}",
        f"Completed runs: {len(results)}",
        "",
        "RESULTS — PERCENTAGES",
        (
            f"{'Seed':<8} {'Best epoch':>12} {'Dev F1':>10} "
            f"{'Precision':>12} {'Recall':>10} {'Test F1':>10}"
        ),
    ]

    for result in results:
        lines.append(
            f"{result['seed']:<8} "
            f"{result['best_epoch']:>12} "
            f"{result['dev_f1']:>10.2f} "
            f"{result['precision']:>12.2f} "
            f"{result['recall']:>10.2f} "
            f"{result['f1']:>10.2f}"
        )

    if len(results) >= 2:
        lines.append("")
        lines.append("MEAN ± SAMPLE STANDARD DEVIATION")

        for metric in ("precision", "recall", "f1"):
            values = [result[metric] for result in results]
            lines.append(
                f"Test {metric}: "
                f"{mean(values):.2f} ± {stdev(values):.2f}"
            )

    summary = "\n".join(lines)

    path = os.path.join(BASE_SAVE_DIR, "five_seed_summary.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(summary)

    return summary, path


seed_results = []

for seed in SEEDS:

    set_seed(seed)

    SAVE_DIR = os.path.join(BASE_SAVE_DIR, f"seed_{seed}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("\n" + "=" * 70)
    print("Model:", BERT_NAME)
    print("Seed:", seed)
    print("Save directory:", SAVE_DIR)
    print("=" * 70)

    train_sents = read_conllu(TRAIN_CONLLU)
    dev_sents = read_conllu(DEV_CONLLU)
    test_sents = read_conllu(TEST_CONLLU)

    labels = [
        tag
        for sent in train_sents
        for _, tag in sent
    ]
    vocab = Vocab(labels)

    print("Train sentences:", len(train_sents))
    print("Dev sentences:", len(dev_sents))
    print("Test sentences:", len(test_sents))
    print("NER classes:", len(vocab.lab2id))
    print(vocab.lab2id)

    tokenizer = AutoTokenizer.from_pretrained(
        BERT_NAME,
        use_fast=True,
    )

    train_ds = NERDataset(train_sents, vocab)
    dev_ds = NERDataset(dev_sents, vocab)
    test_ds = NERDataset(test_sents, vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
    )

    model = BertNER(
        BERT_NAME,
        len(vocab.lab2id),
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)

    steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(steps * 0.1),
        num_training_steps=steps,
    )

    # Ensures the first evaluated checkpoint is always saved.
    best_f1 = float("-inf")
    best_epoch = None
    bad = 0

    best_path = os.path.join(SAVE_DIR, "best_ner.pt")
    label_path = os.path.join(SAVE_DIR, "labels.txt")

    with open(label_path, "w", encoding="utf-8") as f:
        for i in range(len(vocab.id2lab)):
            f.write(vocab.id2lab[i] + "\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for enc, y in tqdm(
            train_loader,
            desc=f"Seed {seed} | Train {epoch}/{EPOCHS}",
        ):
            enc = {k: v.to(device) for k, v in enc.items()}
            y = y.to(device)

            optimizer.zero_grad()
            loss, _ = model(enc, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        dev_p, dev_r, dev_f1, dev_report = evaluate(
            model,
            dev_loader,
        )

        print("\nSeed:", seed)
        print("Epoch:", epoch)
        print("Loss:", total_loss / len(train_loader))
        print("Dev Precision:", dev_p)
        print("Dev Recall:", dev_r)
        print("Dev F1:", dev_f1)

        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            bad = 0

            torch.save(model.state_dict(), best_path)
            print("Saved best model")

        else:
            bad += 1

        if bad >= PATIENCE:
            print("Early stopping")
            break

    model.load_state_dict(
        torch.load(
            best_path,
            map_location=device,
            weights_only=True,
        )
    )

    test_p, test_r, test_f1, test_report = evaluate(
        model,
        test_loader,
    )

    print("\nSeed:", seed)
    print("Best epoch:", best_epoch)
    print("Best Dev F1:", best_f1)
    print("Final Test Precision:", test_p)
    print("Final Test Recall:", test_r)
    print("Final Test F1:", test_f1)
    print("\nClassification Report:\n")
    print(test_report)
    print("Model saved:", best_path)
    print("Labels saved:", label_path)

    report_path = os.path.join(SAVE_DIR, "test_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(
            f"Model: {BERT_NAME}\n"
            f"Seed: {seed}\n"
            f"Best epoch: {best_epoch}\n"
            f"Best Dev F1: {best_f1:.8f}\n"
            f"Test Precision: {test_p:.8f}\n"
            f"Test Recall: {test_r:.8f}\n"
            f"Test F1: {test_f1:.8f}\n\n"
            f"{test_report}"
        )

    seed_results.append({
        "seed": seed,
        "best_epoch": best_epoch,
        "dev_f1": best_f1 * 100,
        "precision": test_p * 100,
        "recall": test_r * 100,
        "f1": test_f1 * 100,
    })

    # Save progress after each completed seed.
    write_summary(seed_results)

    # Release this run's model before starting the next.
    del optimizer, scheduler, model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


summary_text, summary_path = write_summary(seed_results)

print("\n" + "=" * 70)
print(summary_text)
print("\nSummary saved:", summary_path)