'''Training
This code is implemented by Jiawei Wang, wangjiawei531@gmail.com
Usage: python -u train_classifier.py
'''

from utils import SchedulerCosineDecayWarmup, get_precision_recall_f1, evaluate
from model import Classification, Bert_average, BertMatching, BertMatchingDAG, BertMatchingHierarchy
from dataset import CaseData
import argparse
import os
import logging
from copy import deepcopy

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import AdamW, AutoTokenizer
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="LJP")
    parser.add_argument('--batch_size', '-b', type=int,
                        default=24, help='default: 24')
    parser.add_argument('--framework', type=str, default='classification',
                        help='[classification, matching, matching_dag, BertMatchingHierarchy] default: matching')
    parser.add_argument('--model_type', type=str, default='BertCLS',
                        help='[BertCLS, TextCNN] default: BertCLS')
    parser.add_argument('--epochs', type=int,
                        default=10, help='default: 10')
    parser.add_argument('--learning_rate', type=float,
                        default=1e-5, help='default: 1e-5')
    parser.add_argument('--input_max_length', '-l', type=int,
                        default=512, help='default: 512')
    parser.add_argument('--forward_bound', '-fb',
                        action='store_true', help='default: False')
    parser.add_argument('--froze_bert', '-froze',
                        action='store_true', help='default: False')
    parser.add_argument(
        '--resume', '-r', action='store_true', help='default: False')
    parser.add_argument('--train_path', type=str, default='./datasets/cail_small/process_small_train.json',
                        help='default: ./datasets/cail_small/process_small_train.json')
    parser.add_argument('--valid_path', type=str, default='./datasets/cail_small/process_small_valid.json',
                        help='default: ./datasets/cail_small/process_small_valid.json')
    parser.add_argument('--test_path', type=str, default='./datasets/cail_small/process_small_test.json',
                        help='default: ./datasets/cail_small/process_small_test.json')
    parser.add_argument('--checkpoint_path', '-c', type=str, default='./checkpoint/model_matching_hierarchy_best.pth',
                        help='default: ./checkpoint/model_matching_hierarchy_best.pth')
    parser.add_argument('--save_path', '-s', type=str, default='./checkpoint/model_matching_hierarchy_best.pth',
                        help='default: ./checkpoint/model_matching_hierarchy_best.pth')
    args = parser.parse_args()
    logging.info(args)

    # check the device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    logging.info('Using {} device'.format(device))

    torch.cuda.empty_cache()
    torch.manual_seed(888)
    np.random.seed(888)

    # prepare training data
    training_data = CaseData(mode='train', train_file=args.train_path)
    valid_data = CaseData(mode='valid', valid_file=args.valid_path)
    test_data = CaseData(mode='valid', valid_file=args.test_path)

    train_dataloader = DataLoader(
        training_data, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=training_data.collate_function)
    valid_dataloader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=valid_data.collate_function)
    test_dataloader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=test_data.collate_function)

    if 'small' in args.train_path:
        num_classes_law, num_classes_accu, num_classes_term = 103, 119, 11
    elif 'big' in args.train_path:
        num_classes_law, num_classes_accu, num_classes_term = 118, 130, 11
    # load the model and tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.framework == 'classification':
        model = Classification(device, num_classes_accu=num_classes_accu,
                             num_classes_law=num_classes_law, num_classes_term=num_classes_term, args=args)
        logging.info('Load classification framework')
    elif args.framework == 'matching':
        model = BertMatching(device, num_classes_accu=num_classes_accu,
                             num_classes_law=num_classes_law, num_classes_term=num_classes_term)
        logging.info('Load BertMatching')
    elif args.framework == 'matching_dag':
        model = BertMatchingDAG(device, num_classes_accu=num_classes_accu,
                                num_classes_law=num_classes_law, num_classes_term=num_classes_term)
        logging.info('Load BertMatchingDAG')
    elif args.framework == 'BertMatchingHierarchy':
        model = BertMatchingHierarchy(device, num_classes_accu=num_classes_accu,
                                      num_classes_law=num_classes_law, num_classes_term=num_classes_term, args=args)
    else:
        raise NameError
    tokenizer = AutoTokenizer.from_pretrained('./pre_model/bert-base-chinese')

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode='max', factor=0.5, patience=3, verbose=True)  # max for acc
    scheduler = SchedulerCosineDecayWarmup(
        optimizer, lr=args.learning_rate, warmup_len=3, total_iters=args.epochs)

    # resume checkpoint
    if args.resume:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # 'cpu' to 'gpu'
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        logging.info(
            f"Resume model and optimizer from checkpoint '{args.checkpoint_path}' with epoch {checkpoint['epoch']} and best F1 score of {checkpoint['best_f1_score']}")
        logging.info(f"optimizer lr: {optimizer.param_groups[0]['lr']}")
        start_epoch = checkpoint['epoch']
        best_f1_score = checkpoint['best_f1_score']
    else:
        # start training process
        start_epoch = 0
        best_f1_score = 0

    model.to(device)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader):
            facts, labels_accu, labels_law, labels_term = data

            # move data to device
            labels_accu = torch.from_numpy(np.array(labels_accu)).to(device)
            labels_law = torch.from_numpy(np.array(labels_law)).to(device)
            labels_term = torch.from_numpy(np.array(labels_term)).to(device)

            # forward and backward propagations
            optimizer.zero_grad()
            if args.forward_bound:
                loss, loss_forward, logits_accu, logits_law, logits_term, bound_probs_accu, bound_probs_term = model(
                    facts, labels_accu, labels_law, labels_term)
                (loss + loss_forward).backward()
            else:
                loss, logits_accu, logits_law, logits_term = model(
                    facts, labels_accu, labels_law, labels_term)
                loss.backward()
            optimizer.step()

            # logging.info statistics
            alpha = 0.9
            running_loss = alpha * running_loss + (1 - alpha) * loss.item()

            if i % 500 == 0:
                predictions_accu = logits_accu.detach().cpu().numpy()
                labels_accu = labels_accu.cpu().numpy()

                predictions_law = logits_law.detach().cpu().numpy()
                labels_law = labels_law.cpu().numpy()

                predictions_term = logits_term.detach().cpu().numpy()
                labels_term = labels_term.cpu().numpy()
                # logging.info(
                #     f'epoch{epoch+1}, step{i*args.batch_size:6d}/{len(training_data)}, loss: {running_loss:.5f}, loss_mi: {loss_mi:.5f}')
                if args.forward_bound:
                    logging.info(
                        f'epoch{epoch+1}, step{i*args.batch_size:6d}/{len(training_data)}, loss: {running_loss:.5f}, loss_forward: {loss_forward}')
                else:
                    logging.info(
                        f'epoch{epoch+1}, step{i*args.batch_size:6d}/{len(training_data)}, loss: {running_loss:.5f}')

                pred_accu = np.argmax(predictions_accu, axis=1)
                pred_law = np.argmax(predictions_law, axis=1)
                pred_term = np.argmax(predictions_term, axis=1)
                if args.forward_bound:
                    pred_accu_forward_bound = np.argmax(bound_probs_accu.detach().cpu().numpy(), axis=1)
                    pred_term_forward_bound = np.argmax(bound_probs_term.detach().cpu().numpy(), axis=1)

                accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = get_precision_recall_f1(
                    labels_accu, pred_accu, 'macro')
                logging.info(
                    f'train accusation macro accuracy:{accuracy_accu:.5f} precision:{p_macro_accu:.5f}, recall:{r_macro_accu:.5f}, f1_score:{f1_macro_accu:.5f}')
                accuracy_law, p_macro_law, r_macro_law, f1_macro_law = get_precision_recall_f1(
                    labels_law, pred_law, 'macro')
                logging.info(
                    f'train law article macro accuracy:{accuracy_law:.5f} precision:{p_macro_law:.5f}, recall:{r_macro_law:.5f}, f1_score:{f1_macro_law:.5f}')
                accuracy_term, p_macro_term, r_macro_term, f1_macro_term = get_precision_recall_f1(
                    labels_term, pred_term, 'macro')
                logging.info(
                    f'train term of penalty macro accuracy:{accuracy_term:.5f} precision:{p_macro_term:.5f}, recall:{r_macro_term:.5f}, f1_score:{f1_macro_term:.5f}')
                logging.info(
                    f'train average accuracy : {(accuracy_accu + accuracy_law + accuracy_term)/3:.5f}. f1_score: {(f1_macro_accu + f1_macro_law+ f1_macro_term)/3:.5f}')

                if args.forward_bound:
                    accuracy_accu, p_macro_accu, r_macro_accu, f1_macro_accu = get_precision_recall_f1(
                        labels_accu, pred_accu_forward_bound, 'macro')
                    logging.info(
                        f'Forward Bound: train accusation macro accuracy:{accuracy_accu:.5f} precision:{p_macro_accu:.5f}, recall:{r_macro_accu:.5f}, f1_score:{f1_macro_accu:.5f}')
                    accuracy_term, p_macro_term, r_macro_term, f1_macro_term = get_precision_recall_f1(
                        labels_term, pred_term_forward_bound, 'macro')
                    logging.info(
                        f'Forward Bound: train term of penalty macro accuracy:{accuracy_term:.5f} precision:{p_macro_term:.5f}, recall:{r_macro_term:.5f}, f1_score:{f1_macro_term:.5f}')

        if (epoch + 1) % 1 == 0:
            logging.info('Evaluating the model on validation set...')
            f1_macro_accu, f1_macro_law, f1_macro_term = evaluate(
                test_dataloader, tokenizer, model, device, args)
            
            f1_avg = (f1_macro_accu + f1_macro_law + f1_macro_term) / 3.0
            scheduler.step()

            if f1_avg > best_f1_score:
                best_f1_score = f1_avg
                logging.info(
                    f"the valid best average F1 score is {best_f1_score}.")
                state = {
                    'epoch': epoch,
                    'state_dict': deepcopy(model.state_dict()),
                    'optimizer': deepcopy(optimizer.state_dict()),
                    'best_f1_score': best_f1_score,
                }
                save_model_path = args.save_path
                torch.save(state, save_model_path)
                logging.info(f'Save model in path: {save_model_path}')
    
    logging.info('Load best checkpoint for testing model.')
    checkpoint = torch.load(args.save_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    logging.info('Testing Model')
    f1_macro_accu, f1_macro_law, f1_macro_term = evaluate(
        test_dataloader, tokenizer, model, device, args)
    f1_avg = (f1_macro_accu + f1_macro_law + f1_macro_term) / 3.0
    logging.info(f'average f1 score: {f1_avg}')
