
'''
Usage: python -u main_verifier.py
'''
from utils import SchedulerCosineDecayWarmup, get_precision_recall_f1
from model import Verifier
from dataset import CaseData
import argparse
import os
import logging
from copy import deepcopy
from collections import defaultdict
from concurrent import futures
import codecs
import json

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AdamW, AutoTokenizer
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)


def multi_cjs_evaluate(args, device, test_dataloader, model):
    # Evaluation
    model.eval()

    all_labels_accu = []
    all_labels_law = []
    all_labels_term = []
    all_verifier_logits = []
    all_path_batch = []
    count = 0
    for i, data in enumerate(test_dataloader):
        facts, labels_accu, labels_law, labels_term = data

        # move data to device
        labels_accu = torch.from_numpy(np.array(labels_accu)).to(device)
        labels_law = torch.from_numpy(np.array(labels_law)).to(device)
        labels_term = torch.from_numpy(np.array(labels_term)).to(device)

        # forward
        with torch.no_grad():
            verifier_logits, path_batch = model(
                facts, is_train=False)

        all_labels_accu.append(labels_accu.cpu())
        all_labels_law.append(labels_law.cpu())
        all_labels_term.append(labels_term.cpu())
        all_verifier_logits.append(verifier_logits.detach().cpu())
        all_path_batch.append(path_batch.detach().cpu())
        count = count + 1
        if count % 100 == 0:
            print(f'{count * args.batch_size} samples have been processed.')
        # if i > 100:
        #     break

    all_labels_accu = torch.cat(all_labels_accu, dim=0).numpy()
    all_labels_law = torch.cat(all_labels_law, dim=0).numpy()
    all_labels_term = torch.cat(all_labels_term, dim=0).numpy()
    all_verifier_logits = torch.cat(all_verifier_logits, dim=0)
    all_path_batch = torch.cat(all_path_batch, dim=0)

    def topk_evaluation(all_path_batch, all_verifier_logits, topk):
        '''
        all_path_batch: [B, beam_size**3, 3]
        all_verifier_logits: [B, beam_size**3, 3]
        '''
        logging.info(f'=== Top{topk} evaluation ===')
        
        pred_path_batch = []
        for path_per_sample, logits_per_sample in zip(all_path_batch, all_verifier_logits):
            # path_per_sample: [beam_size**3, 3]
            # logits_per_sample: [beam_size**3, 3]
            _, topk_indices = torch.topk(logits_per_sample.mean(1), k=topk, dim=0)  # [topk]

            index_prob_law = defaultdict(torch.FloatTensor)
            index_prob_accu = defaultdict(torch.FloatTensor)
            index_prob_term = defaultdict(torch.FloatTensor)

            # Get votes of probs for each label of different types.
            path_per_sample = path_per_sample.numpy().tolist()
            for index in topk_indices:
                index_prob_law.setdefault(path_per_sample[index][0], torch.FloatTensor([0]))  # add(key, value)
                index_prob_accu.setdefault(path_per_sample[index][1], torch.FloatTensor([0]))
                index_prob_term.setdefault(path_per_sample[index][2], torch.FloatTensor([0]))

                index_prob_law[path_per_sample[index][0]] += logits_per_sample[index][0]
                index_prob_accu[path_per_sample[index][1]] += logits_per_sample[index][1]
                index_prob_term[path_per_sample[index][2]] += logits_per_sample[index][2]

            # Get votes of probs for each candidate sample.
            path_per_candidate_sample_votes = []
            for index in topk_indices:
                sum_votes_from_each_type = index_prob_law[path_per_sample[index][0]] + \
                    index_prob_accu[path_per_sample[index][1]] + \
                    index_prob_term[path_per_sample[index][2]]
                path_per_candidate_sample_votes.append([path_per_sample[index], sum_votes_from_each_type])

            # Sorted the candidate samples via its votes.
            path_per_candidate_sample_votes = sorted(path_per_candidate_sample_votes, key=lambda x: x[1], reverse=True)

            # # Select best candidate sample.
            pred_path_per_sample = path_per_candidate_sample_votes[0][0]

            # law_index = sorted(index_prob_law.items(), key= lambda x: x[1], reverse=True)[0][0]
            # accu_index = sorted(index_prob_accu.items(), key= lambda x: x[1], reverse=True)[0][0]
            # term_index = sorted(index_prob_term.items(), key= lambda x: x[1], reverse=True)[0][0]
            # pred_path_per_sample = [law_index, accu_index, term_index]

            # Gathering the prediction of each sample.
            pred_path_batch.append(np.array(pred_path_per_sample))

        pred_path_batch = np.stack(pred_path_batch, axis=0)

        pred_law, pred_accu, pred_term = pred_path_batch[:, 0], pred_path_batch[:, 1], pred_path_batch[:, 2]

        accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = get_precision_recall_f1(
            all_labels_accu, pred_accu, 'macro')
        logging.info(
            f'test accusation macro accuracy:{accuracy_accu:.5f} precision:{p_macro_accu:.5f}, recall:{r_macro_accu:.5f}, f1_score:{f1_macro_accu:.5f}')
        accuracy_law, p_macro_law, r_macro_law, f1_macro_law = get_precision_recall_f1(
            all_labels_law, pred_law, 'macro')
        logging.info(
            f'test law article macro accuracy:{accuracy_law:.5f} precision:{p_macro_law:.5f}, recall:{r_macro_law:.5f}, f1_score:{f1_macro_law:.5f}')
        accuracy_term, p_macro_term, r_macro_term, f1_macro_term = get_precision_recall_f1(
            all_labels_term, pred_term, 'macro')
        logging.info(
            f'test term of penalty macro accuracy:{accuracy_term:.5f} precision:{p_macro_term:.5f}, recall:{r_macro_term:.5f}, f1_score:{f1_macro_term:.5f}')
        logging.info(
            f'test average accuracy : {(accuracy_accu + accuracy_law + accuracy_term)/3:.5f}. f1_score: {(f1_macro_accu + f1_macro_law+ f1_macro_term)/3:.5f}')

        # Check
        if topk == 1:
            pred_path_batch = torch.stack([path_per_sample[index] for path_per_sample, index in zip(
                all_path_batch, all_verifier_logits.mean(2).max(1).indices)]).numpy()  # [b, 3]
            pred_law, pred_accu, pred_term = pred_path_batch[:,
                                                             0], pred_path_batch[:, 1], pred_path_batch[:, 2]
            accuracy_accu_old, p_macro_accu_old, r_macro_accu_old, f1_macro_accu_old = get_precision_recall_f1(
                all_labels_accu, pred_accu, 'macro')

            assert accuracy_accu == accuracy_accu_old, 'Unmatch between old and new implementation.'
            assert p_macro_accu == p_macro_accu_old, 'Unmatch between old and new implementation.'
            assert r_macro_accu == r_macro_accu_old, 'Unmatch between old and new implementation.'
            assert f1_macro_accu == f1_macro_accu_old, 'Unmatch between old and new implementation.'
   
        return topk, f1_macro_accu, f1_macro_law, f1_macro_term
        #topk_f1_avg.append([topk, (f1_macro_accu + f1_macro_law + f1_macro_term)/3])

    topk_f1_avg = []
    # topk_list = [1, 2, 3, 5, 7, 10, 15] + list(range(20, (args.beam_size**2)*11, 20)) + [(args.beam_size**2)*11]
    # topk_list = [1] + list(range(100, 200, 20)) + [(args.beam_size**2)*11]
    # topk_list = [1] + list(range(0, 275, 25))[1:] + [(args.beam_size**2)*11]
    topk_list = [10, 20, 40, 60, 80, 120, 140, 160, 180, 220]
    print('topk_list===:', topk_list)
    # Multi Process
    # with futures.ProcessPoolExecutor(max_workers=16) as executor:
    #     futures_list = [executor.submit(topk_evaluation(all_path_batch, all_verifier_logits, topk))for topk in topk_list]
            
    #     for result_from_future in futures.as_completed(futures_list):
    #         #f1_avg = result_from_future.result()
    #         #topk_f1_avg.append(f1_avg)
    #         topk, f1_macro_accu, f1_macro_law, f1_macro_term = result_from_future.result()
    #         topk_f1_avg.append([topk, (f1_macro_accu + f1_macro_law + f1_macro_term)/3])

    # Single Process
    for topk in topk_list:
        logging.info(f'=== Top{topk} evaluation ===')
        topk, f1_macro_accu, f1_macro_law, f1_macro_term = topk_evaluation(
            all_path_batch, all_verifier_logits, topk=topk)
        topk_f1_avg.append([topk, (f1_macro_accu + f1_macro_law + f1_macro_term)/3])
    best_topk, best_f1_avg = sorted(topk_f1_avg, key=lambda x: x[1], reverse=True)[0]

    # best_topk, best_f1_avg = sorted(
    #     list(zip(topk_list, topk_f1_avg)), key=lambda x: x[1], reverse=True)[0]
    logging.info(
        f'The best result appears in top{best_topk}, best_f1_avg:{best_f1_avg}')

    return best_f1_avg


def cjs_topk_evaluate(args, device, test_dataloader, model, topk):
    '''with cjs, topk high-likelihood label combinations for a given law case.'''
    # Evaluation
    model.eval()

    all_labels_accu = []
    all_labels_law = []
    all_labels_term = []
    all_verifier_logits = []
    all_path_batch = []
    count = 0
    for i, data in enumerate(test_dataloader):
        facts, labels_accu, labels_law, labels_term = data

        # move data to device
        labels_accu = torch.from_numpy(np.array(labels_accu)).to(device)
        labels_law = torch.from_numpy(np.array(labels_law)).to(device)
        labels_term = torch.from_numpy(np.array(labels_term)).to(device)

        # forward
        with torch.no_grad():
            verifier_logits, path_batch = model(
                facts, is_train=False)

        all_labels_accu.append(labels_accu.cpu())
        all_labels_law.append(labels_law.cpu())
        all_labels_term.append(labels_term.cpu())
        all_verifier_logits.append(verifier_logits.detach().cpu())
        all_path_batch.append(path_batch.detach().cpu())
        count = count + 1
        if count % 100 == 0:
            print(f'{count * args.batch_size} samples have been processed.')

    all_labels_accu = torch.cat(all_labels_accu, dim=0).numpy()
    all_labels_law = torch.cat(all_labels_law, dim=0).numpy()
    all_labels_term = torch.cat(all_labels_term, dim=0).numpy()
    all_verifier_logits = torch.cat(all_verifier_logits, dim=0)
    all_path_batch = torch.cat(all_path_batch, dim=0)

    def topk_evaluation(all_path_batch, all_verifier_logits, topk):
        '''
        all_path_batch: [B, beam_size**3, 3]
        all_verifier_logits: [B, beam_size**3, 3]
        '''
        logging.info(f'=== Top{topk} evaluation ===')
        # output to file
        fw = codecs.open(args.pred_save_path, 'w', encoding='utf-8')
    
        for path_per_sample, logits_per_sample in zip(all_path_batch, all_verifier_logits):
            # path_per_sample: [beam_size**3, 3]
            # logits_per_sample: [beam_size**3, 3]
            _, topk_indices = torch.topk(logits_per_sample.mean(1), k=topk, dim=0)  # [topk]

            index_prob_law = defaultdict(torch.FloatTensor)
            index_prob_accu = defaultdict(torch.FloatTensor)
            index_prob_term = defaultdict(torch.FloatTensor)

            # Get votes of probs for each label of different types.
            # TODO: sum or average
            path_per_sample = path_per_sample.numpy().tolist()
            for index in topk_indices:
                index_prob_law.setdefault(path_per_sample[index][0], torch.FloatTensor([0]))  # add(key, value)
                index_prob_accu.setdefault(path_per_sample[index][1], torch.FloatTensor([0]))
                index_prob_term.setdefault(path_per_sample[index][2], torch.FloatTensor([0]))

                index_prob_law[path_per_sample[index][0]] += logits_per_sample[index][0]
                index_prob_accu[path_per_sample[index][1]] += logits_per_sample[index][1]
                index_prob_term[path_per_sample[index][2]] += logits_per_sample[index][2]
            
            # Get votes of probs for each candidate sample.
            path_per_candidate_sample_votes = []
            for index in topk_indices:
                sum_votes_from_each_type = index_prob_law[path_per_sample[index][0]] + \
                    index_prob_accu[path_per_sample[index][1]] + \
                    index_prob_term[path_per_sample[index][2]]
                path_per_candidate_sample_votes.append([path_per_sample[index], sum_votes_from_each_type])

            # Sorted the candidate samples via its votes.
            path_per_candidate_sample_votes = sorted(path_per_candidate_sample_votes, key=lambda x: x[1], reverse=True)

            # Select topk best candidate sample.
            output_dict = {}
            for i in range(8):
                label_combination, label_combination_probs = path_per_candidate_sample_votes[i][0], path_per_candidate_sample_votes[i][1].cpu().numpy()[0]
                # law charge term
                output_dict['top_' + str(i+1)] = str(label_combination[0]) + '-' + str(label_combination[1]) + '-' + str(label_combination[2]) + '-' + str(round(float(label_combination_probs),4))
            
            fw.write(json.dumps(output_dict) + '\n')
        fw.close()

    topk_evaluation(all_path_batch, all_verifier_logits, topk=topk)


def topk_evaluate(args, device, test_dataloader, model, topk):
    '''w/o cjs, topk high-likelihood label combinations for a given law case.'''
    # Evaluation
    model.eval()

    all_labels_accu = []
    all_labels_law = []
    all_labels_term = []
    all_verifier_logits = []
    all_path_batch = []
    count = 0
    for i, data in enumerate(test_dataloader):
        facts, labels_accu, labels_law, labels_term = data

        # move data to device
        labels_accu = torch.from_numpy(np.array(labels_accu)).to(device)
        labels_law = torch.from_numpy(np.array(labels_law)).to(device)
        labels_term = torch.from_numpy(np.array(labels_term)).to(device)

        # forward
        with torch.no_grad():
            verifier_logits, path_batch = model(
                facts, is_train=False)
 
        all_labels_accu.append(labels_accu.cpu())
        all_labels_law.append(labels_law.cpu())
        all_labels_term.append(labels_term.cpu())
        all_verifier_logits.append(verifier_logits.detach().cpu())
        all_path_batch.append(path_batch.detach().cpu())
        count = count + 1
        if count % 100 == 0:
            print(f'{count * args.batch_size} samples have been processed.')
    
    all_labels_accu = torch.cat(all_labels_accu, dim=0).numpy()
    all_labels_law = torch.cat(all_labels_law, dim=0).numpy()
    all_labels_term = torch.cat(all_labels_term, dim=0).numpy()
    all_verifier_logits = torch.cat(all_verifier_logits, dim=0)
    all_path_batch = torch.cat(all_path_batch, dim=0)
    
    # output to file
    fw = codecs.open(args.pred_save_path, 'w', encoding='utf-8')

    for path_per_sample, logits_per_sample in zip(all_path_batch, all_verifier_logits):
        topk_vals, topk_indices = torch.topk(logits_per_sample.mean(1), k=topk, dim=0)  # [topk]
        output_dict = {}
        for id, (index, pro) in enumerate(zip(topk_indices, topk_vals)):
            label_combination = '-'.join([str(i) for i in path_per_sample[index].cpu().numpy()]) + '-' + str(round(float(pro.cpu().numpy()), 4))
            output_dict['top_' + str(id+1)] = label_combination
        fw.write(json.dumps(output_dict) + '\n')
    fw.close()
        
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="LJP")
    parser.add_argument('--batch_size', '-b', type=int,
                        default=32, help='default: 24')
    parser.add_argument('--input_max_length', '-l', type=int,
                        default=512, help='default: 512')
    parser.add_argument('--model_type', type=str, default='BertCLS',
                        help='[BertCLS, TextCNN] default: BertCLS')
    parser.add_argument('--beam_size', '-beam', type=int,
                        default=5, help='default: 5')
    parser.add_argument('--forward_bound', '-fb',
                        action='store_true', help='default: False')
    parser.add_argument('--froze_bert', '-froze',
                        action='store_true', help='default: False')
    parser.add_argument('--multi_label', '-ml',
                        action='store_true', help='default: False')
    parser.add_argument('--train_path', type=str, default='./datasets/cail_small/process_small_train.json',
                        help='default: ./datasets/cail_small/process_small_train.json')
    parser.add_argument('--valid_path', type=str, default='./datasets/cail_small/process_small_valid.json',
                        help='default: ./datasets/cail_small/process_small_valid.json')
    parser.add_argument('--test_path', type=str, default='./datasets/cail_small/process_small_test.json',
                        help='default: ./datasets/cail_small/process_small_test.json')
    parser.add_argument('--save_path', '-s', type=str, default='./checkpoint/model_seoncd_stage_verifier_best.pth',
                        help='default: ./checkpoint/model_seoncd_stage_verifier_best.pth')
    parser.add_argument('--pred_save_path', '-pred', type=str, default='./results/small_verifier_pred.json',
                        help='default: ./results/small_verifier_pred.json')
    parser.add_argument('--evaluate_mode', '-em', type=str, default='evaluate',
                        help='[multi_cjs_evaluate, cjs_topk_evaluate, topk_evaluate]')
    args = parser.parse_args()
    logging.info(args)

    # check the device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    logging.info('Using {} device'.format(device))

    torch.cuda.empty_cache()
    torch.manual_seed(2021)
    np.random.seed(2021)

    # prepare training data
    training_data = CaseData(mode='train', train_file=args.train_path)
    valid_data = CaseData(mode='valid', valid_file=args.test_path)
    test_data = CaseData(mode='valid', valid_file=args.test_path)

    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=training_data.collate_function, drop_last=True)
    valid_dataloader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=valid_data.collate_function)
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=test_data.collate_function)

    # load the model and tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if 'small' in args.train_path:
        num_classes_law, num_classes_accu, num_classes_term = 103, 119, 11
    elif 'big' in args.train_path:
        num_classes_law, num_classes_accu, num_classes_term = 118, 130, 11

    model = Verifier(device, num_classes_accu=num_classes_accu,
                     num_classes_law=num_classes_law, num_classes_term=num_classes_term, args=args)

    logging.info('Load best checkpoint for testing model.')
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    if args.evaluate_mode == 'multi_cjs_evaluate':
        # 1.test
        logging.info('Testing Model')
        f1_avg = multi_cjs_evaluate(args, device, test_dataloader, model)
        logging.info(f'average f1 score: {f1_avg}')
    elif args.evaluate_mode == 'cjs_topk_evaluate':
        # 2.output topk high-likelihood label combinations
        cjs_topk_evaluate(args, device, test_dataloader, model, topk=120)
    elif args.evaluate_mode== 'topk_evaluate':
        topk_evaluate(args, device, test_dataloader, model, topk=8)
        
