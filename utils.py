import logging
import torch
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from collections import defaultdict


def labels_to_multihot(labels, num_classes=146):
    multihot_labels = torch.zeros(len(labels), num_classes)
    for i, label in enumerate(labels):
        for l in label:
            multihot_labels[i][l] = 1
    return multihot_labels


def get_precision_recall_f1(y_true: np.array, y_pred: np.array, average='micro'):
    precision = metrics.precision_score(
        y_true, y_pred, average=average, zero_division=0)
    recall = metrics.recall_score(
        y_true, y_pred, average=average, zero_division=0)
    f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy, precision, recall, f1


def get_precision_recall_f1_curve(y_true: np.array, y_pred: np.array):
    result = defaultdict(list)
    for threshold in np.linspace(0, 0.5, 51):
        result['threshold'].append(threshold)
        tmp_y_pred = np.array(y_pred >= threshold, np.int64)

        p, r, f1_micro = get_precision_recall_f1(y_true, tmp_y_pred, 'micro')
        result['precision_micro'].append(round(p, 4))
        result['recall_micro'].append(round(r, 4))
        result['f1_micro'].append(round(f1_micro, 4))

        p, r, f1_macro = get_precision_recall_f1(y_true, tmp_y_pred, 'macro')
        result['precision_macro'].append(round(p, 4))
        result['recall_macro'].append(round(r, 4))
        result['f1_macro'].append(round(f1_macro, 4))

        result['f1_avg'].append(round((f1_micro+f1_macro)/2, 4))
    return result


def evaluate(valid_dataloader, tokenizer, model, device, args):
    model.eval()
    all_predictions_accu = []
    all_predictions_law = []
    all_predictions_term = []
    all_labels_accu = []
    all_labels_law = []
    all_labels_term = []
    if args.forward_bound:
        all_predictions_accu_forward_bound = []
        all_predictions_term_forward_bound = []
    for i, data in enumerate(valid_dataloader):
        facts, labels_accu, labels_law, labels_term = data

        # move data to device
        labels_accu = torch.from_numpy(np.array(labels_accu)).to(device)
        labels_law = torch.from_numpy(np.array(labels_law)).to(device)
        labels_term = torch.from_numpy(np.array(labels_term)).to(device)
     
        with torch.no_grad():
            # forward
            if args.forward_bound:
                logits_accu, logits_law, logits_term, bound_probs_accu, bound_probs_term = model(
                    facts)
            else:
                logits_accu, logits_law, logits_term = model(facts)

        all_labels_accu.append(labels_accu.cpu())
        all_labels_law.append(labels_law.cpu())
        all_labels_term.append(labels_term.cpu())

        all_predictions_accu.append(logits_accu.detach().cpu())
        all_predictions_law.append(logits_law.detach().cpu())
        all_predictions_term.append(logits_term.detach().cpu())
        if args.forward_bound:
            all_predictions_accu_forward_bound.append(bound_probs_accu.detach().cpu())
            all_predictions_term_forward_bound.append(bound_probs_term.detach().cpu())

    all_predictions_accu = torch.cat(all_predictions_accu, dim=0).numpy()
    all_labels_accu = torch.cat(all_labels_accu, dim=0).numpy()
    all_predictions_law = torch.cat(all_predictions_law, dim=0).numpy()
    all_labels_law = torch.cat(all_labels_law, dim=0).numpy()
    all_predictions_term = torch.cat(all_predictions_term, dim=0).numpy()
    all_labels_term = torch.cat(all_labels_term, dim=0).numpy()
    if args.forward_bound:
        all_predictions_accu_forward_bound = torch.cat(all_predictions_accu_forward_bound, dim=0).numpy()
        all_predictions_term_forward_bound = torch.cat(all_predictions_term_forward_bound, dim=0).numpy()
    
    print(all_labels_accu.shape, np.argmax(all_predictions_accu, axis=1).shape)
    accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = get_precision_recall_f1(all_labels_accu, np.argmax(all_predictions_accu, axis=1), 'macro')
    accuracy_law, p_macro_law, r_macro_law, f1_macro_law = get_precision_recall_f1(all_labels_law, np.argmax(all_predictions_law, axis=1), 'macro')
    accuracy_term, p_macro_term, r_macro_term, f1_macro_term = get_precision_recall_f1(all_labels_term, np.argmax(all_predictions_term, axis=1), 'macro')

    logging.info(f'accusation macro accuracy:{accuracy_accu:.4f} precision:{p_macro_accu:.4f}, recall:{r_macro_accu:.4f}, f1_score:{f1_macro_accu:.4f}')
    logging.info(f'law article macro accuracy:{accuracy_law:.4f} precision:{p_macro_law:.4f}, recall:{r_macro_law:.4f}, f1_score:{f1_macro_law:.4f}')
    logging.info(f'term of penalty macro accuracy:{accuracy_term:.4f} precision:{p_macro_term:.4f}, recall:{r_macro_term:.4f}, f1_score:{f1_macro_term:.4f}')

    if args.forward_bound:
        accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = get_precision_recall_f1(
            all_labels_accu, np.argmax(all_predictions_accu_forward_bound, axis=1), 'macro')
        logging.info(
            f'Forward Bound: accusation macro accuracy:{accuracy_accu:.4f} precision:{p_macro_accu:.4f}, recall:{r_macro_accu:.4f}, f1_score:{f1_macro_accu:.4f}')
        accuracy_term, p_macro_term, r_macro_term, f1_macro_term = get_precision_recall_f1(
            all_labels_term, np.argmax(all_predictions_term_forward_bound, axis=1), 'macro')
        logging.info(
            f'Forward Bound: term of penalty macro accuracy:{accuracy_term:.4f} precision:{p_macro_term:.4f}, recall:{r_macro_term:.4f}, f1_score:{f1_macro_term:.4f}')
    return f1_macro_accu, f1_macro_law, f1_macro_term


class SchedulerCosineDecayWarmup:
    def __init__(self, optimizer, lr, warmup_len, total_iters, current_iter=0, verbose=True):
        self.optimizer = optimizer
        self.lr = lr
        self.warmup_len = warmup_len
        self.total_iters = total_iters
        self.current_iter = current_iter
        self.verbose = verbose
        self.step()
    
    def get_lr(self):
        if self.current_iter < self.warmup_len:
            lr = self.lr * (self.current_iter + 1) / self.warmup_len
        else:
            cur = self.current_iter - self.warmup_len
            total= self.total_iters - self.warmup_len
            lr = 0.5 * (1 + np.cos(np.pi * cur / total)) * self.lr
        return lr
    
    def step(self):
        lr = self.get_lr()
        for param in self.optimizer.param_groups:
            param['lr'] = lr
        self.current_iter += 1
        if self.verbose:
            print(f'Learning rate has been set to {lr}.')