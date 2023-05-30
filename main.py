import random
import pandas as pd
import os
from sklearn.model_selection import KFold
import copy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from torch import nn
from torch.optim import Adam
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
import torch
from models import *
import numpy as np
from dataloader import *
import argparse
# import warnings
# warnings.filterwarnings("ignore")

def run(model, train_dataloader, val_dataloader, test_dataloader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    model = model.to(device)
    best_kappa = 0
    best_f1 = 0

    for epoch_num in range(args.epochs):
        total_acc_train = 0
        total_loss_train = 0
        for train_essay_id, train_token, train_label in train_dataloader:
            train_label = train_label.to(device)

            train_mask_token = train_token['attention_mask'].to(device)
            train_input_ids_token = train_token['input_ids'].squeeze(1).to(device)

            logits = model(train_input_ids_token, train_mask_token)
            outputs = torch.sigmoid(logits)
            batch_loss = criterion(logits, train_label.unsqueeze(1).float())
            total_loss_train += batch_loss.item()

            preds = outputs.reshape(-1).round()
            acc = (preds == train_label.reshape(-1)).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()
        scheduler.step()

        # validation set
        total_acc_val = 0
        total_loss_val = 0

        gold_labels = []
        pred_labels = []
        essay_ids = []
        with torch.no_grad():
            for epoch, (val_essay_id, val_token, val_label) in enumerate(val_dataloader):
                val_label = val_label.to(device)

                val_mask_token = val_token['attention_mask'].to(device)
                val_input_ids_tokens = val_token['input_ids'].squeeze(1).to(device)

                logits = model(val_input_ids_tokens, val_mask_token)
                outputs = torch.sigmoid(logits)
                preds = outputs.reshape(-1).round()

                batch_loss = criterion(logits, val_label.unsqueeze(1).float())
                total_loss_val += batch_loss.item()

                acc = (preds == val_label.reshape(-1)).sum().item()
                total_acc_val += acc

                gold_labels += val_label.cpu().numpy().tolist()
                pred_labels += preds.cpu().numpy().tolist()
                essay_ids += val_essay_id.cpu().numpy().tolist()

            kappa = cohen_kappa_score(gold_labels, pred_labels, weights="quadratic")
            f1 = classification_report(gold_labels, pred_labels, output_dict=True)["macro avg"]["f1-score"]
            if f1 > best_f1:
                best_kappa = kappa
                best_f1 = f1

                # test set
                total_acc_test = 0
                total_loss_test = 0

                gold_labels = []
                pred_labels = []
                essay_ids = []
                with torch.no_grad():
                    for test_essay_id, test_token, test_label in test_dataloader:
                        test_label = test_label.to(device)

                        test_mask_token = test_token['attention_mask'].to(device)
                        test_input_ids_tokens = test_token['input_ids'].squeeze(1).to(device)

                        logits = model(test_input_ids_tokens, test_mask_token)
                        outputs = torch.sigmoid(logits)
                        preds = outputs.reshape(-1).round()

                        batch_loss = criterion(logits, test_label.unsqueeze(1).float())
                        total_loss_test += batch_loss.item()

                        acc = (preds == test_label.reshape(-1)).sum().item()
                        total_acc_test += acc

                        gold_labels += test_label.cpu().numpy().tolist()
                        pred_labels += preds.cpu().numpy().tolist()
                        essay_ids += test_essay_id.cpu().numpy().tolist()

                    test_kappa = cohen_kappa_score(gold_labels, pred_labels, weights="quadratic")
                    test_f1 = classification_report(gold_labels, pred_labels, output_dict=True)["macro avg"]["f1-score"]

                    fold_test_gold_label = gold_labels
                    fold_test_pred_label = pred_labels
                    fold_test_essay_id = essay_ids


        print(f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_dataset): .3f} \
                | Train Accuracy: {total_acc_train / len(train_dataset): .3f} \
                | Val Loss: {total_loss_val / len(val_dataset): .3f} \
                | Val Accuracy: {total_acc_val / len(val_dataset): .3f} \
                | Val Kappa: {best_kappa: .3f} \
                | Val F1: {best_f1: .3f} \
                | Test Kappa {test_kappa: .3f} \
                | Test F1 {test_f1: .3f}')

    return fold_test_essay_id, fold_test_gold_label, fold_test_pred_label


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='revision quality classification')
    parser.add_argument('--exp-dir', default='./experiments/debug',
                        help='path to the experiment')
    parser.add_argument('--exp-type', default='success', choices=["success", "purpose"],
                        help='classify success or purpose labels')
    parser.add_argument('--data-source', default="college", choices=["mixture", "college"],
                        help='data source')
    parser.add_argument('--purpose-type', default="everything", choices=["reasoning", "evidence", "everything"],
                        help='specify fine labels')
    parser.add_argument('--context-type', default="claim_summary",
                        choices=["neighbor-short", "neighbor-long", "reasoning", "evidence", "claim", "claim_summary", "reasoning_summary", "evidence_summary"],
                        help='specify context of sentences')
    parser.add_argument('--epochs', default=10, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch-size', default=32, type=int,
                        help='mini-batch size')
    parser.add_argument('--learning-rate', default=5e-5, type=float,
                        help='initial learning rate')
    parser.add_argument('--step-size', default=4, type=int,
                        help='decay-step number')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='decay-step factor')
    parser.add_argument('--random-seed', default=22, type=int,
                        help='random seed')
    parser.add_argument('--k-fold', default=5, type=int,
                        help='k fold cross validation')
    parser.add_argument('--model-name', default="distilroberta-base",
                        help='model name')
    parser.add_argument('--model-type', default="baseline", choices=["baseline"],
                        help='model type')
    parser.add_argument('--freeze-model', default='false', type=lambda s: s.lower() in ['true', "True", "TRUE"],
                        help='freeze inner layers of the model')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    np.random.seed(22)
    torch.manual_seed(args.random_seed)
    random.seed(22)
    if not os.path.exists(args.exp_dir):
        os.mkdir(args.exp_dir)

    df = get_master_data_revision_purpose(args)

    if args.purpose_type == "reasoning":
        master_df = df[df["fine_labels"] == "reasoning"]
    elif args.purpose_type == "evidence":
        master_df = df[df["fine_labels"] == "evidence"]
    elif args.purpose_type == "everything":
        master_df = df
    else:
        raise "wrong set!"
    master_df = master_df.reset_index(drop=True)
    master_df = master_df.sample(frac=1, random_state=22).reset_index(drop=True)


    initial_model = DesirableModel(args)


    kappa_list = []
    avg_precision_list = []
    avg_recall_list = []
    avg_f1_list = []
    desirable_f1_list = []
    undesirable_f1_list = []
    gold_label_list = []
    pred_label_list = []
    essay_id_list = []
    fold_df_list = []

    kfold = KFold(n_splits=args.k_fold, shuffle=True, random_state=22)
    for fold, (train_val_idx, test_idx) in enumerate(kfold.split(master_df)):
        print('Running fold {}...'.format(fold + 1))

        k1 = int(len(train_val_idx)*0.8)
        train_idx = train_val_idx[:k1]
        val_idx = train_val_idx[k1:]

        if len(train_idx) % args.batch_size == 1:  # in case last batch is 1 channel
            train_idx = np.append(train_idx, train_idx[0])
        if len(val_idx) % args.batch_size == 1:  # in case last batch is 1 channel
            val_idx = np.append(val_idx, val_idx[0])
        if len(test_idx) % args.batch_size == 1:  # in case last batch is 1 channel
            test_idx = np.append(test_idx, test_idx[0])

        train_data = master_df.iloc[train_idx].reset_index()
        val_data = master_df.iloc[val_idx].reset_index()
        test_data = master_df.iloc[test_idx].reset_index()

        # if args.data_source == "college":
        #     train_data = get_augment_data(train_data)
        #     val_data = get_augment_data(val_data)

        fold_df_list.append(test_data)

        train_dataset = DesirableDataset(args, train_data)
        val_dataset = DesirableDataset(args, val_data)
        test_dataset = DesirableDataset(args, test_data)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn = my_collate)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = my_collate)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn = my_collate)

        model = copy.deepcopy(initial_model)
        essay_id, gold_label, pred_label = run(model, train_dataloader, val_dataloader, test_dataloader)

        kappa = cohen_kappa_score(gold_label, pred_label, weights="quadratic")
        report = classification_report(gold_label, pred_label, output_dict=True)
        avg_precision = report["macro avg"]["precision"]
        avg_recall = report["macro avg"]["recall"]

        avg_f1 = report["macro avg"]["f1-score"]
        desirable_f1 = report["1"]["f1-score"]
        undesirable_f1 = report["0"]["f1-score"]

        kappa_list.append(kappa)

        avg_precision_list.append(avg_precision)
        avg_recall_list.append(avg_recall)

        avg_f1_list.append(avg_f1)
        desirable_f1_list.append(desirable_f1)
        undesirable_f1_list.append(undesirable_f1)

        gold_label_list += gold_label
        pred_label_list += pred_label
        essay_id_list += essay_id

        print(classification_report(gold_label, pred_label))
        print("WQK:", kappa)
        print("fold-level confusion matrix:")
        print(confusion_matrix(gold_label, pred_label))

    fold_df = pd.concat(fold_df_list, ignore_index=False)
    fold_df["pred_score"] = pred_label_list
    fold_df["gold_score"] = gold_label_list
    fold_df = fold_df.reset_index(drop=True)
    fold_df.to_csv(os.path.join(args.exp_dir, "results.csv"))

    avg_kappa = round(np.mean(kappa_list), 4) * 100
    avg_precision = round(np.mean(avg_precision_list), 4) * 100
    avg_recall = round(np.mean(avg_recall_list), 4) * 100
    avg_f1 = round(np.mean(avg_f1_list), 4) * 100
    desirable_f1 = round(np.mean(desirable_f1_list), 4) * 100
    undesirable_f1 = round(np.mean(undesirable_f1_list), 4) * 100
    print("----------")
    print('{:<15}{:<15}{:<15}{:<15}{:<15}{:<15}'.format("desirable_f1", "undesirable_f1", "avg_precision", "avg_recall", "avg_f1", "avg_kappa"))
    print('{:<15.2f}{:<15.2f}{:<15.2f}{:<15.2f}{:<15.2f}{:<15.2f}'.format(desirable_f1, undesirable_f1, avg_precision, avg_recall, avg_f1, avg_kappa))
    print("overall confusion matrix:")
    print(confusion_matrix(gold_label_list, pred_label_list))
