'''
python -u statistic_topk.py -fb -c=./checkpoint/model_cls_froze6_epoch5_best.pth
'''
from pickle import NONE
import numpy as np
from numpy import average
import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
import torch.nn.functional as F
from utils import labels_to_multihot, get_precision_recall_f1
from model import Classification, Bert_average, BertMatching, BertMatchingDAG, BertMatchingHierarchy, Verifier
from dataset import CaseData
import argparse
import os
import logging
from copy import deepcopy

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AdamW, AutoTokenizer
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

parser = argparse.ArgumentParser(description="LJP")
parser.add_argument('--model_type', type=str, default='BertCLS',
                        help='[TextCNN, BertCLS, NeurJudge] default: BertCLS')
parser.add_argument('--batch_size', '-b', type=int,
                    default=64, help='default: 8')
parser.add_argument('--input_max_length', '-l', type=int,
                    default=512, help='default: 512')
parser.add_argument('--beam_size', '-beam', type=int,
                    default=1, help='default: 1')
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
args = parser.parse_args()
logging.info(args)

# check the device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
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
    training_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=training_data.collate_function)
valid_dataloader = DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=valid_data.collate_function)
test_dataloader = DataLoader(
    test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_data.collate_function)

# load the model and tokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"

num_classes_law = 103
num_classes_accu = 119
num_classes_term = 11

model = Classification(device, num_classes_accu=num_classes_accu,
                             num_classes_law=num_classes_law, num_classes_term=num_classes_term, args=args)

tokenizer = AutoTokenizer.from_pretrained('./pre_model/bert-base-chinese')


# resume checkpoint
checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])
print(f'load model {args.checkpoint_path}')
# 'cpu' to 'gpu'

logging.info(
    f"Resume model and optimizer from checkpoint '{args.checkpoint_path}' with epoch {checkpoint['epoch']} and best F1 score of {checkpoint['best_f1_score']}")

model.to(device)

class AccMeter():
    def __init__(self, topk=1) -> None:
        self.topk = topk
        self.law_positive = 0
        self.accu_positive = 0
        self.term_positive = 0
        self.total_num = 0
    
    def compute(self, logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term):
        indices_topk_law = torch.topk(logits_law, k=self.topk, dim=1).indices
        indices_topk_accu = torch.topk(logits_accu, k=self.topk, dim=1).indices
        indices_topk_term = torch.topk(logits_term, k=self.topk, dim=1).indices

        batch_size = logits_law.shape[0]
        for i in range(batch_size):
            law_gt = labels_law[i]
            accu_gt = labels_accu[i]
            term_gt = labels_term[i]

            law_pred = indices_topk_law[i]
            accu_pred = indices_topk_accu[i]
            term_pred = indices_topk_term[i]

            if law_gt in law_pred:
                self.law_positive += 1
            if accu_gt in accu_pred:
                self.accu_positive += 1
            if term_gt in term_pred:
                self.term_positive += 1
            self.total_num += 1
        
    def print(self):
        print(f'top{self.topk}: law_positive:{self.law_positive}, total_num:{self.total_num}, top{self.topk}_acc: {self.law_positive/self.total_num}')
        print(f'top{self.topk}: accu_positive:{self.accu_positive}, total_num:{self.total_num}, top{self.topk}_acc: {self.accu_positive/self.total_num}')
        print(f'top{self.topk}: term_positive:{self.term_positive}, total_num:{self.total_num}, top{self.topk}_acc: {self.term_positive/self.total_num}')

AccMeter_top1 = AccMeter(topk=1)
# AccMeter_top2 = AccMeter(topk=2)
# AccMeter_top3 = AccMeter(topk=3)
# AccMeter_top4 = AccMeter(topk=4)
# AccMeter_top5 = AccMeter(topk=5)
# AccMeter_top6 = AccMeter(topk=6)
# AccMeter_top7 = AccMeter(topk=7)
# AccMeter_top8 = AccMeter(topk=8)
# AccMeter_top9 = AccMeter(topk=9)
# AccMeter_top10 = AccMeter(topk=10)
# AccMeter_top11 = AccMeter(topk=11)

model.eval()
for batch_idx, data in enumerate(test_dataloader):
    facts, labels_accu, labels_law, labels_term = data

    # move data to device
    if labels_accu is not None and labels_law is not None and labels_term is not None:
        labels_accu = torch.from_numpy(np.array(labels_accu)).to(device)
        labels_law = torch.from_numpy(np.array(labels_law)).to(device)
        labels_term = torch.from_numpy(np.array(labels_term)).to(device)
    with torch.no_grad():
        #logits_accu, logits_law, logits_term, output_law, output_accu, output_term, case_embeddings = self.model(inputs, rt_outputs=True)
        #logits_accu, logits_law, logits_term = model(facts, rt_outputs=True)
        _, logits_law, _, logits_accu, logits_term = model(facts)

    AccMeter_top1.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)
    # AccMeter_top2.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)
    # AccMeter_top3.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)
    # AccMeter_top4.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)
    # AccMeter_top5.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)
    # AccMeter_top6.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)
    # AccMeter_top7.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)
    # AccMeter_top8.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)
    # AccMeter_top9.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)
    # AccMeter_top10.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)
    # AccMeter_top11.compute(logits_law, logits_accu, logits_term, labels_law, labels_accu, labels_term)

    # path_batch = []
    # path_label_batch = []
    # for batch_idx, (indices_topk_law_, indices_topk_accu_, indices_topk_term_) in enumerate(zip(indices_topk_law, indices_topk_accu, indices_topk_term)):
    #     path_per_sample = []
    #     path_label_per_sample = []
    #     for index_law in indices_topk_law_:
    #         path = []
    #         path.append(index_law)
    #         for index_accu in indices_topk_accu_:
    #             path = path[:1]
    #             path.append(index_accu)
    #             for index_term in indices_topk_term_:
    #                 path = path[:2]
    #                 path.append(index_term)
    #                 # path_per_sample.append(torch.cat(path))
    #                 path_per_sample.append(torch.LongTensor(path))
                    
    #                 tmp_path_label_per_sample = []
    #                 # TODO: when the gold label can't search by top-k beam search, add the glod triples via gold label (only for training mode). 
    #                 if int(labels_law[batch_idx].cpu().numpy()) == int(index_law.cpu().numpy()):
    #                     tmp_path_label_per_sample.append(torch.LongTensor([1]))
    #                 else:
    #                     tmp_path_label_per_sample.append(torch.LongTensor([0]))
                    
    #                 if int(labels_accu[batch_idx].cpu().numpy()) == int(index_accu.cpu().numpy() - num_classes_law):
    #                     tmp_path_label_per_sample.append(torch.LongTensor([1]))
    #                 else:
    #                     tmp_path_label_per_sample.append(torch.LongTensor([0]))
                    
    #                 if int(labels_term[batch_idx].cpu().numpy()) == int(index_term.cpu().numpy() - num_classes_law - num_classes_accu):
    #                     tmp_path_label_per_sample.append(torch.LongTensor([1]))
    #                 else:
    #                     tmp_path_label_per_sample.append(torch.LongTensor([0]))
    #                 path_label_per_sample.append(torch.cat(tmp_path_label_per_sample))

    #     path_batch.append(torch.cat(path_per_sample))
    #     path_label_batch.append(torch.cat(path_label_per_sample))

    # #path_batch = torch.cat(path_batch).view(B * self.args.beam_size**3, 3).to(self.device)  # [b * self.args.beam_size**3, 3]
    # path_batch = torch.cat(path_batch).view(B, args.beam_size**3, 3).to(device)  # [b, self.args.beam_size**3, 3]

    # path_label_batch = torch.stack(path_label_batch)   # [b, self.args.beam_size**3, 3]

    if (batch_idx) % 5000 == 0:
        AccMeter_top1.print()
        # AccMeter_top2.print()
        # AccMeter_top3.print()
        # AccMeter_top4.print()
        # AccMeter_top5.print()
        # AccMeter_top6.print()
        # AccMeter_top7.print()
        # AccMeter_top8.print()
        # AccMeter_top9.print()
        # AccMeter_top10.print()
        # AccMeter_top11.print()
        print('===')

print('===End===')
AccMeter_top1.print()
# AccMeter_top2.print()
# AccMeter_top3.print()
# AccMeter_top4.print()
# AccMeter_top5.print()
# AccMeter_top6.print()
# AccMeter_top7.print()
# AccMeter_top8.print()
# AccMeter_top9.print()
# AccMeter_top10.print()
# AccMeter_top11.print()
