'''
Usage: python -u main_classifier.py
'''
from cProfile import label
import codecs
import sys
import re
import random
import argparse
import os
from turtle import st

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
import numpy as np
import json

from dataset import CaseData
from model import Classification
from utils import labels_to_multihot, get_precision_recall_f1
import logging


def evaluation(args, device, test_dataloader, model):
    all_predictions_accu = []
    all_predictions_law = []
    all_predictions_term = []
    all_labels_accu = []
    all_labels_law = []
    all_labels_term = []
    model.eval()
    for i, data in enumerate(test_dataloader):
        facts, labels_accu, labels_law, labels_term = data

        # move data to device
        labels_accu = torch.from_numpy(np.array(labels_accu)).to(device)
        labels_law = torch.from_numpy(np.array(labels_law)).to(device)
        labels_term = torch.from_numpy(np.array(labels_term)).to(device)

        # forward and backward propagations
        with torch.no_grad():
            logits_accu, logits_law, logits_term = model(facts, rt_outputs=True)

        all_predictions_accu.append(logits_accu.softmax(dim=1).detach().cpu())
        all_labels_accu.append(labels_accu.cpu())
        all_predictions_law.append(logits_law.softmax(dim=1).detach().cpu())
        all_labels_law.append(labels_law.cpu())
        all_predictions_term.append(logits_term.softmax(dim=1).detach().cpu())
        all_labels_term.append(labels_term.cpu())

        if i % (args.batch_size * 100) == 0:
            print(f'Processing sample {i * args.batch_size }/{len(test_dataloader) * args.batch_size}')

    all_predictions_accu = torch.cat(all_predictions_accu, dim=0).numpy()
    all_labels_accu = torch.cat(all_labels_accu, dim=0).numpy()
    all_predictions_law = torch.cat(all_predictions_law, dim=0).numpy()
    all_labels_law = torch.cat(all_labels_law, dim=0).numpy()
    all_predictions_term = torch.cat(all_predictions_term, dim=0).numpy()
    all_labels_term = torch.cat(all_labels_term, dim=0).numpy()

    all_predictions_accu = np.argmax(all_predictions_accu, axis=1)
    all_predictions_law = np.argmax(all_predictions_law, axis=1)
    all_predictions_term = np.argmax(all_predictions_term, axis=1)

    accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = get_precision_recall_f1(
        all_labels_accu, all_predictions_accu, 'macro')
    accuracy_law, p_macro_law, r_macro_law, f1_macro_law = get_precision_recall_f1(
        all_labels_law, all_predictions_law, 'macro')
    accuracy_term, p_macro_term, r_macro_term, f1_macro_term = get_precision_recall_f1(
        all_labels_term, all_predictions_term, 'macro')

    # output
    print(
        f'Test: accusation macro accuracy:{accuracy_accu:.4f} precision:{p_macro_accu:.4f}, recall:{r_macro_accu:.4f}, f1_score:{f1_macro_accu:.4f}')
    print(
        f'Test: law article macro accuracy:{accuracy_law:.4f} precision:{p_macro_law:.4f}, recall:{r_macro_law:.4f}, f1_score:{f1_macro_law:.4f}')
    print(
        f'Test: term of penalty macro accuracy:{accuracy_term:.4f} precision:{p_macro_term:.4f}, recall:{r_macro_term:.4f}, f1_score:{f1_macro_term:.4f}')
    f1_avg = (f1_macro_accu + f1_macro_law + f1_macro_term) / 3.0
    print(f'Test: Average F1 score: {f1_avg}')
    
    # output to file
    fw = codecs.open(args.pred_save_path, 'w', encoding='utf-8')
    for pred_law, pred_accu, pred_term in zip(all_predictions_law, all_predictions_accu, all_predictions_term):
        pred_dict = {"pred_law": str(pred_law),
                     "pred_accu": str(pred_accu),
                     "pred_term": str(pred_term)}
        fw.write(json.dumps(pred_dict) + '\n')
    fw.close()


def topk_evaluation(args, test_dataloader, model, law_topk=1, accu_topk=1, term_topk=1):
    model.eval()
    total_num = len(test_dataloader) * args.batch_size
    combination_correct_num = 0
    # output to file
    fw = codecs.open(args.pred_save_path, 'w', encoding='utf-8')
    for i, data in enumerate(test_dataloader):
        facts, labels_accu, labels_law, labels_term = data

        labels_accu = np.array(labels_accu)
        labels_law = np.array(labels_law)
        labels_term = np.array(labels_term)

        # forward and backward propagations
        with torch.no_grad():
            #logits_accu, logits_law, logits_term = model(facts, rt_outputs=True)
            _, logits_law, _, logits_accu, logits_term = model(facts)
        
        probs_law = logits_law.softmax(dim=1).detach()
        probs_accu = logits_accu.softmax(dim=1).detach()
        probs_term = logits_term.softmax(dim=1).detach()
        
        val_topk_law, idx_topk_law = torch.topk(probs_law, law_topk)
        val_topk_accu, idx_topk_accu = torch.topk(probs_accu, accu_topk)
        val_topk_term, idx_topk_term = torch.topk(probs_term, term_topk)

        val_topk_law, idx_topk_law = val_topk_law.cpu().numpy(), idx_topk_law.cpu().numpy()
        val_topk_accu, idx_topk_accu = val_topk_accu.cpu().numpy(), idx_topk_accu.cpu().numpy()
        val_topk_term, idx_topk_term = val_topk_term.cpu().numpy(), idx_topk_term.cpu().numpy()
        
        for idx_law, idx_accu, idx_term, val_law, val_accu, val_term, label_law, label_accu, label_term in zip(idx_topk_law, idx_topk_accu, idx_topk_term, val_topk_law, val_topk_accu, val_topk_term, labels_law, labels_accu, labels_term):
            pred_label_combinations = []
            for idx_law_, val_law_ in zip(idx_law, val_law):
                for idx_accu_, val_accu_ in zip(idx_accu, val_accu):
                    for idx_term_, val_term_ in zip(idx_term, val_term):
                        label_combination = str(idx_law_) + '-' + str(idx_accu_) + '-' + str(idx_term_)
                        label_combination_prob = (val_law_ + val_accu_ + val_term_) / 3.0
                        pred_label_combinations.append([label_combination, label_combination_prob])
            # sort by average probs
            pred_label_combinations = sorted(pred_label_combinations, key=lambda x: x[1], reverse=True)

            if i % (args.batch_size * 10) == 0:
                print(f'Processing sample {i * args.batch_size }/{len(test_dataloader) * args.batch_size}')

            if pred_label_combinations[0][0] == str(label_law) + '-' + str(label_accu) + '-' + str(label_term):
                combination_correct_num += 1
                output_dict = {'top1_isTrue': 1}
            else:
                output_dict = {'top1_isTrue': 0}
            
            for i in range(8):
                output_dict['top_' + str(i+1)] = pred_label_combinations[i][0] + '-' + str(round(float(pred_label_combinations[i][1]), 5))
            
            fw.write(json.dumps(output_dict) + '\n')
    fw.close()
    print(f'Test combinations accuracy: {combination_correct_num}/{total_num}={combination_correct_num / total_num}.')
 

def topk_accuracy_evaluation(args, test_dataloader, model, total_num):
    model.eval()
    topk_list = [1, 3, 5, 7, 9, 11]
    for topk in topk_list:
        law_correct_count = 0
        accu_correct_count = 0
        term_correct_count = 0
        for i, data in enumerate(test_dataloader):
            facts, labels_accu, labels_law, labels_term = data

            labels_accu = np.array(labels_accu)
            labels_law = np.array(labels_law)
            labels_term = np.array(labels_term)

            # forward and backward propagations
            with torch.no_grad():
                #logits_accu, logits_law, logits_term = model(facts, rt_outputs=True)
                _, logits_law, _, logits_accu, logits_term = model(facts)
            
            probs_law = logits_law.softmax(dim=1).detach()
            probs_accu = logits_accu.softmax(dim=1).detach()
            probs_term = logits_term.softmax(dim=1).detach()

            val_topk_law, idx_topk_law = torch.topk(probs_law, topk)
            val_topk_accu, idx_topk_accu = torch.topk(probs_accu, topk)
            val_topk_term, idx_topk_term = torch.topk(probs_term, topk)

            val_topk_law, idx_topk_law = val_topk_law.cpu().numpy(), idx_topk_law.cpu().numpy()
            val_topk_accu, idx_topk_accu = val_topk_accu.cpu().numpy(), idx_topk_accu.cpu().numpy()
            val_topk_term, idx_topk_term = val_topk_term.cpu().numpy(), idx_topk_term.cpu().numpy()

        
            for idx_law, idx_accu, idx_term, label_law, label_accu, label_term in zip(idx_topk_law, idx_topk_accu, idx_topk_term, labels_law, labels_accu, labels_term):
                if label_law in idx_law:
                    law_correct_count += 1
                if label_accu in idx_accu:
                    accu_correct_count += 1
                if label_term in idx_term:
                    term_correct_count += 1
        print(f'<***{topk} accuracy***>')
        print(f'law article:{law_correct_count/total_num}. accu:{accu_correct_count/total_num}. term:{term_correct_count/total_num}.')


def main():
    parser = argparse.ArgumentParser(description="LJP")
    parser.add_argument('--model_type', type=str, default='BertCLS',
                            help='[TextCNN, BertCLS, NeurJudge] default: BertCLS')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=32, help='default: 8')
    parser.add_argument('--input_max_length', '-l', type=int,
                        default=512, help='default: 512')
    parser.add_argument('--forward_bound', '-fb',
                        action='store_true', help='default: False')
    parser.add_argument('--froze_bert', '-froze',
                        action='store_true', help='default: False')
    parser.add_argument('--train_path', type=str, default='./datasets/cail_small/process_small_train.json',
                        help='default: ./datasets/cail_small/process_small_train.json')
    parser.add_argument('--valid_path', type=str, default='./datasets/cail_small/process_small_valid.json',
                        help='default: ./datasets/cail_small/process_small_valid.json')
    parser.add_argument('--test_path', type=str, default='./datasets/cail_small/process_small_test.json',
                        help='default: ./datasets/cail_small/process_small_test.json')
    parser.add_argument('--checkpoint_path', '-c', type=str, default='./checkpoint/model_cls_froze6_epoch5_best.pth',
                        help='default: ./checkpoint/model_cls_froze6_epoch5_best.pth')
    parser.add_argument('--pred_save_path', '-pred', type=str, default='./results/pred.txt',
                        help='default: ./results/pred.json')
    parser.add_argument('--evaluate_mode', '-em', type=str, default='evaluate',
                        help='[evaluate, topk_evaluate]')
    args = parser.parse_args()
    logging.info(args)

    # check the device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    test_data = CaseData(mode='valid', valid_file=args.test_path)
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_data.collate_function)

    # load the model and tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = Classification(device, num_classes_accu=119,
                             num_classes_law=103, num_classes_term=11, args=args).to(device)
    print('Load Bert_average')

    # resume checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    print(f'load model {args.checkpoint_path}')
    # 'cpu' to 'gpu'

    logging.info(
        f"Resume model and optimizer from checkpoint '{args.checkpoint_path}' with epoch {checkpoint['epoch']} and best F1 score of {checkpoint['best_f1_score']}")

    model.to(device)
    if args.evaluate_mode == 'evaluate':
        evaluation(args, device, test_dataloader, model)
    elif args.evaluate_mode == 'topk_evaluate':
        # output topk high-likelihood label combinations
        topk_evaluation(args, test_dataloader, model, law_topk=3, accu_topk=3, term_topk=3)
    elif args.evaluate_mode == 'topk_accuracy':
        # calculate the top accuracy of different kinds of labels
        topk_accuracy_evaluation(args, test_dataloader, model, len(test_data))
        

if __name__ == '__main__':
    main()
