from pickle import NONE
import numpy as np
import json
from numpy import average
import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
import torch.nn.functional as F
from neurjudge.model import NeurJudge
from neurjudge.utils import Data_Process
from transformers import AdamW, AutoTokenizer

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.lstm = nn.LSTM(768, 768, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.5)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


class TextCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(TextCNN, self).__init__()
        filter_sizes = (2, 3, 4)
        self.convs = nn.ModuleList(
            [nn.Conv1d(768, 256, kernel_size=k) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2))
        return x

    def forward(self, x):
        out = x.transpose(1, 2) # B, C, N
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1) # B, C, N
        out = self.dropout(out)
        return out.squeeze(2)


class TextRCNN(nn.Module):
    def __init__(self):
        super(TextRCNN, self).__init__()
        self.rnn = TextRNN()
        self.cnn = TextCNN()

    def forward(self, x):
        return self.cnn(self.rnn(x))


class NeurJudgeEnc(nn.Module):
    def __init__(self, device, args):
        super(NeurJudgeEnc, self).__init__()
        self.device = device
        self.process = Data_Process()
        word2id = json.load(open('./neurjudge/data/cail_small/small_word2id.json'))
        word_embed_path = './neurjudge/data/cail_small/small_w2v.txt'
        word_embedding = self.process.load_vectors(word_embed_path)
        embedding = self.process.get_embedding_matrix(word2id, word_embedding)

        self.model = NeurJudge(torch.from_numpy(embedding))
        
        self.model.load_state_dict(torch.load(args.checkpoint_path))
        self.model.to(device)
        print(f'Load NeurJudge Model: {args.checkpoint_path}')

        self.legals, self.legals_len,self.arts,self.arts_sent_lent,self.charge_tong2id,self.id2charge_tong,self.art2id,self.id2art = self.process.get_graph()

        self.legals,self.legals_len,self.arts,self.arts_sent_lent = self.legals.to(device),self.legals_len.to(device),self.arts.to(device),self.arts_sent_lent.to(device)
        
    def forward(self, facts):
        # preprocess
        fact_all = []
        for fact in facts:
            fact_all.append(self.process.parse(fact))
        documents, sent_lent = self.process.seq2tensor(fact_all, max_len=350)

        documents = documents.to(self.device)
        sent_lent = sent_lent.to(self.device)

        charge_out,article_out,time_out = self.model(self.legals,self.legals_len,self.arts,self.arts_sent_lent,self.charge_tong2id,self.id2charge_tong,self.art2id,self.id2art,documents,sent_lent,self.process,self.device)

        return charge_out, article_out, time_out


class Classification(nn.Module):
    def __init__(self, device, num_classes_accu=None, num_classes_law=None, num_classes_term=None, args=None):
        super(Classification, self).__init__()
        self.device = device
        self.num_classes_accu = num_classes_accu
        self.num_classes_law = num_classes_law
        self.num_classes_term = num_classes_term
        self.args = args

        if self.args.model_type != 'NeurJudge':
            self.backbone = AutoModel.from_pretrained('./pre_model/bert-base-chinese')
            self.tokenizer = AutoTokenizer.from_pretrained('./pre_model/bert-base-chinese')

        if args.model_type == 'TextCNN':
            self.encoder = TextCNN()
        elif args.model_type == 'TextRNN':
            self.encoder = TextRNN()
        elif args.model_type == 'TextRCNN':
            self.encoder = TextRCNN()
        elif args.model_type == 'NeurJudge':
            self.encoder = NeurJudgeEnc(device=device, args=args)
        print(f'model type: {args.model_type}')

        self.linear_accu = nn.Linear(in_features=768, out_features=num_classes_accu)
        self.linear_law = nn.Linear(in_features=768, out_features=num_classes_law)
        self.linear_term = nn.Linear(in_features=768, out_features=num_classes_term)

        if args.forward_bound:
            # DAG mask
            if 'big' in args.train_path:
                law_accu_edges = np.loadtxt('./datasets/law_accu_edges_threshold1_big.txt', dtype='int')
                accu_term_edges = np.loadtxt('./datasets/accu_term_edges_big.txt', dtype='int')
            elif 'small' in args.train_path:
                law_accu_edges = np.loadtxt('./datasets/law_accu_edges.txt', dtype='int')
                accu_term_edges = np.loadtxt('./datasets/accu_term_edges.txt', dtype='int')

            # positions with ``True`` is not allowed to attend in Transformer block
            # while ``False`` values will be unchanged.

            dag_mask = torch.zeros([num_classes_law, num_classes_accu])
            for i, j in law_accu_edges:
                dag_mask[i, j] = 1
            self.dag_mask_l2 = dag_mask.bool()

            dag_mask = torch.zeros([num_classes_accu, num_classes_term])
            for i, j in accu_term_edges:
                dag_mask[i, j] = 1
            self.dag_mask_l3 = dag_mask.bool()

        self.CE_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()

        if self.args.model_type != 'NeurJudge' and args.froze_bert:
            self.trainable_param_names = ['layer.11', 'layer.10', 'layer.9', 'layer.8', 'layer.7', 'layer.6']
            for name, param in self.backbone.named_parameters():
                if any(n in name for n in self.trainable_param_names):
                    param.requires_grad = True
                else:
                    param.requires_grad = False


    def forward(self, facts=None, labels_accu=None, labels_law=None, labels_term=None, rt_outputs=None):
        # move data to device
        if self.args.model_type == 'NeurJudge':
            with torch.no_grad():
                logits_accu, logits_law, logits_term = self.encoder(facts)
                
        else:
            # tokenize the data text
            inputs = self.tokenizer(list(facts), max_length=512,
                               padding=True, truncation=True, return_tensors='pt')

            input_ids = inputs['input_ids'].to(self.device)
            token_type_ids = inputs['token_type_ids'].to(self.device)
            attention_mask = inputs['attention_mask'].to(self.device)

            if self.args.model_type == 'BertCLS':
                outputs = self.backbone(input_ids, attention_mask, token_type_ids)
                pooler_output = outputs['pooler_output']
            else:
                with torch.no_grad():
                    outputs = self.backbone(input_ids, attention_mask, token_type_ids)
                pooler_output = self.encoder(outputs['last_hidden_state'])
                
            logits_accu = self.linear_accu(pooler_output)
            logits_law = self.linear_law(pooler_output)
            logits_term = self.linear_term(pooler_output)

            # prob_accu = logits_accu.softmax(dim=1)
            # prob_law = logits_law.softmax(dim=1)
            # prob_term = logits_term.softmax(dim=1)

        if rt_outputs:
            return logits_accu, logits_law, logits_term

        if self.args.forward_bound:
            # [B, Length]
            probs_law = torch.softmax(logits_law, dim=1)
            probs_accu = torch.softmax(logits_accu, dim=1)
            probs_term = torch.softmax(logits_term, dim=1)

            # [B, Length_accu] = [B, Length_law], [Length_law, Length_accu]
            gate_accu = torch.matmul(probs_law, self.dag_mask_l2.to(self.device).float())
            bound_probs_accu = self.normalize_probs(probs_accu * gate_accu)

            # [B, Length_term] = [B, Length_accu], [Length_accu, Length_term]
            gate_term = torch.matmul(probs_accu, self.dag_mask_l3.to(self.device).float())
            bound_probs_term = self.normalize_probs(probs_term * gate_term)

        if self.training:
            loss_law = self.CE_loss(logits_law, labels_law)
            loss_accu = self.CE_loss(logits_accu, labels_accu)
            loss_term = self.CE_loss(logits_term, labels_term)

            loss = loss_law + loss_accu + loss_term

            if self.args.forward_bound:
                loss_accu_forword = self.nll_loss(torch.log(bound_probs_accu + 1e-12), labels_accu)
                loss_term_forword = self.nll_loss(torch.log(bound_probs_term + 1e-12), labels_term)

                loss_forward = loss_accu_forword + loss_term_forword
                return loss, loss_forward, logits_accu, logits_law, logits_term, bound_probs_accu, bound_probs_term
            else:
                return loss, logits_accu, logits_law, logits_term

        if self.args.forward_bound:
            return logits_accu, logits_law, logits_term, bound_probs_accu, bound_probs_term
        else:
            return logits_accu, logits_law, logits_term

    def normalize_probs(self, probs):
        return probs / probs.sum(1, keepdim=True)


class Bert_average(nn.Module):
    def __init__(self, device, num_classes_accu=None, num_classes_law=None, num_classes_term=None):
        super(Bert_average, self).__init__()
        self.model = AutoModel.from_pretrained('./pre_model/bert-base-chinese')
        self.device = device
        self.num_classes_accu = num_classes_accu
        self.num_classes_law = num_classes_law
        self.num_classes_term = num_classes_term

        self.linear_accu = nn.Linear(in_features=768, out_features=num_classes_accu)
        self.linear_law = nn.Linear(in_features=768, out_features=num_classes_law)
        self.linear_term = nn.Linear(in_features=768, out_features=num_classes_term)

        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, inputs=None, labels_accu=None, labels_law=None, labels_term=None, focal_loss=None):
        # move data to device
        input_ids = inputs['input_ids'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask, token_type_ids)
        pooler_output = outputs['last_hidden_state']
        pooler_output = torch.mean(pooler_output, dim=1)

        logits_accu = self.linear_accu(pooler_output)
        logits_law = self.linear_law(pooler_output)
        logits_term = self.linear_term(pooler_output)

        if labels_accu is not None:
            if focal_loss:
                pass
            else:
                loss_accu = self.CE_loss(logits_accu, labels_accu)
                loss_law = self.CE_loss(logits_law, labels_law)
                loss_term = self.CE_loss(logits_term, labels_term)
                loss = loss_accu + loss_law + loss_term
            return loss, logits_accu, logits_law, logits_term

        return logits_accu, logits_law, logits_term


class BertMatching(nn.Module):
    def __init__(self, device, num_classes_accu=None, num_classes_law=None, num_classes_term=None):
        super(BertMatching, self).__init__()
        self.model = AutoModel.from_pretrained('./pre_model/bert-base-chinese')
        self.device = device
        self.num_classes_accu = num_classes_accu
        self.num_classes_law = num_classes_law
        self.num_classes_term = num_classes_term

        # Label Embedding
        self.accu_embedding = nn.Parameter(torch.zeros(num_classes_accu, 768))
        nn.init.kaiming_uniform_(self.accu_embedding, mode='fan_in')
        self.law_embedding = nn.Parameter(torch.zeros(num_classes_law, 768))
        nn.init.kaiming_uniform_(self.law_embedding, mode='fan_in')
        self.term_embedding = nn.Parameter(torch.zeros(num_classes_term, 768))
        nn.init.kaiming_uniform_(self.term_embedding, mode='fan_in')

        self.segment_embedding = nn.Embedding(3, 768)

        # Different information alignment
        self.case_embedding_transform = nn.Linear(768, 768)

        # Relation
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)  # batch_first?
        self.relation_model = nn.TransformerEncoder(transformer_encoder_layer, num_layers=3)

        # Predict
        self.linear_accu = nn.Linear(in_features=768, out_features=1)
        self.linear_law = nn.Linear(in_features=768, out_features=1)
        self.linear_term = nn.Linear(in_features=768, out_features=1)

        # Loss
        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, inputs=None, labels_accu=None, labels_law=None, labels_term=None, focal_loss=None):
        # move data to device
        input_ids = inputs['input_ids'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        outputs = self.model(input_ids, attention_mask, token_type_ids)
        pooler_output = outputs['last_hidden_state']
        case_embedding = torch.mean(pooler_output, dim=1, keepdim=True)  # [B, 1, 768]
        # TODO: Sentence Whitening

        # Transformer
        node_embedding = torch.cat([self.accu_embedding, self.law_embedding, self.term_embedding])
        segment_id = torch.cat([torch.ones(self.num_classes_accu).long()*0, torch.ones(self.num_classes_law).long()*1, torch.ones(self.num_classes_term).long()*2]).to(self.device)
        segment_embedding = self.segment_embedding(segment_id).unsqueeze(0)

        input_embedding = self.case_embedding_transform(case_embedding) + segment_embedding + node_embedding  # [B, L, 768]

        relation_output = self.relation_model(input_embedding)
        output_accu, output_law, output_term = self.unpacked_according_to_three_types_of_labels(relation_output)

        logits_accu = self.linear_accu(output_accu).squeeze(2)
        logits_law = self.linear_law(output_law).squeeze(2)
        logits_term = self.linear_term(output_term).squeeze(2)

        if labels_accu is not None:
            if focal_loss:
                pass
            else:
                loss_accu = self.CE_loss(logits_accu, labels_accu)
                loss_law = self.CE_loss(logits_law, labels_law)
                loss_term = self.CE_loss(logits_term, labels_term)
                loss = loss_accu + loss_law + loss_term
            return loss, logits_accu, logits_law, logits_term

        return logits_accu, logits_law, logits_term

    def unpacked_according_to_three_types_of_labels(self, input):
        return input[:, :self.num_classes_accu], input[:, self.num_classes_accu:self.num_classes_accu+self.num_classes_law], input[:, self.num_classes_accu+self.num_classes_law:self.num_classes_accu+self.num_classes_law+self.num_classes_term]


class BertMatchingDAG(nn.Module):
    def __init__(self, device, num_classes_accu=None, num_classes_law=None, num_classes_term=None):
        super(BertMatchingDAG, self).__init__()
        self.model = AutoModel.from_pretrained('./pre_model/bert-base-chinese')
        self.device = device
        self.num_classes_accu = num_classes_accu
        self.num_classes_law = num_classes_law
        self.num_classes_term = num_classes_term
        num_classes = num_classes_law+num_classes_accu+num_classes_term

        # Label Embedding
        self.law_embedding = nn.Parameter(torch.zeros(num_classes_law, 768))
        nn.init.kaiming_uniform_(self.law_embedding, mode='fan_in')
        self.accu_embedding = nn.Parameter(torch.zeros(num_classes_accu, 768))
        nn.init.kaiming_uniform_(self.accu_embedding, mode='fan_in')
        self.term_embedding = nn.Parameter(torch.zeros(num_classes_term, 768))
        nn.init.kaiming_uniform_(self.term_embedding, mode='fan_in')

        self.segment_embedding = nn.Embedding(3, 768)

        # Different information alignment
        self.case_embedding_transform = nn.Linear(768, 768)

        # Relation
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.relation_model = nn.TransformerEncoder(transformer_encoder_layer, num_layers=3)

        # DAG mask
        law_accu_edges = np.loadtxt('./datasets/law_accu_edges.txt', dtype='int')
        accu_term_edges = np.loadtxt('./datasets/accu_term_edges.txt', dtype='int')

        # positions with ``True`` is not allowed to attend in Transformer block
        # while ``False`` values will be unchanged.
        
        dag_mask = 1 - torch.eye(num_classes)
        for i, j in law_accu_edges:
            dag_mask[i, num_classes_law+j] = 0
        for i, j in accu_term_edges:
            dag_mask[num_classes_law+i, num_classes_law+num_classes_accu+j] = 0
        self.dag_mask = dag_mask.bool().transpose(0, 1)

        # Predict
        self.linear_accu = nn.Linear(in_features=768, out_features=1)
        self.linear_law = nn.Linear(in_features=768, out_features=1)
        self.linear_term = nn.Linear(in_features=768, out_features=1)

        # Loss
        self.CE_loss = nn.CrossEntropyLoss()

    def forward(self, inputs=None, labels_accu=None, labels_law=None, labels_term=None, focal_loss=None):
        # move data to device
        input_ids = inputs['input_ids'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # pooler_output = outputs['last_hidden_state']
        case_embedding = outputs['pooler_output'].unsqueeze(1)

        # case_embedding = torch.mean(pooler_output, dim=1, keepdim=True)  # [B, 1, 768] TODO: max?
        # TODO: Sentence Whitening

        # Transformer
        node_embedding = torch.cat([self.law_embedding, self.accu_embedding, self.term_embedding])
        segment_id = torch.cat([torch.ones(self.num_classes_law).long()*0, torch.ones(self.num_classes_accu).long()*1, torch.ones(self.num_classes_term).long()*2]).to(self.device)
        segment_embedding = self.segment_embedding(segment_id).unsqueeze(0)

        input_embedding = self.case_embedding_transform(case_embedding) + segment_embedding + node_embedding.unsqueeze(0)  # [B, L, 768]

        # attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
        # source sequence length.
        relation_output = self.relation_model(input_embedding, mask=self.dag_mask.to(self.device))
        output_law, output_accu, output_term = self.unpacked_according_to_three_types_of_labels(relation_output)

        logits_law = self.linear_law(output_law).squeeze(2)
        logits_accu = self.linear_accu(output_accu).squeeze(2)
        logits_term = self.linear_term(output_term).squeeze(2)

        if labels_law is not None:
            if focal_loss:
                pass
            else:
                alpha = 3
                loss_law = self.CE_loss(logits_law, labels_law)
                loss_accu = self.CE_loss(logits_accu, labels_accu)
                loss_term = self.CE_loss(logits_term, labels_term)
                # TODO: multiply loss_term by 3 to fasten the optimization?
                loss = loss_law + loss_accu + alpha * loss_term
                # loss_mi = self.mutual_information_loss(relation_output, self.dag_mask.to(self.device))
            return loss, logits_accu, logits_law, logits_term

        return logits_accu, logits_law, logits_term

    def unpacked_according_to_three_types_of_labels(self, input):
        return input[:, :self.num_classes_law], input[:, self.num_classes_law:self.num_classes_law+self.num_classes_accu], input[:, self.num_classes_law+self.num_classes_accu:self.num_classes_law+self.num_classes_accu+self.num_classes_term]

    def mutual_information_loss(self, V, dag_mask):
        # V: shape [B, L, C]
        relation = torch.matmul(V, V.transpose(1, 2))

        # The label "1" indicates the edge between nodes. 
        # TODO: only first order relation, consider two order relation?
        directed_relation_mask = ~dag_mask
        undirected_relation_mask = directed_relation_mask + directed_relation_mask.transpose(0, 1)

        mi_loss = F.binary_cross_entropy_with_logits(relation, undirected_relation_mask.unsqueeze(0).expand_as(relation).float())
        return mi_loss


class BertMatchingHierarchy(nn.Module):
    def __init__(self, device, num_classes_accu=None, num_classes_law=None, num_classes_term=None, args=None):
        super(BertMatchingHierarchy, self).__init__()
        self.model = AutoModel.from_pretrained('./pre_model/bert-base-chinese')
        self.device = device
        self.num_classes_accu = num_classes_accu
        self.num_classes_law = num_classes_law
        self.num_classes_term = num_classes_term
        self.args = args
        num_classes = num_classes_law+num_classes_accu+num_classes_term

        # Label Embedding
        self.law_embedding = nn.Parameter(torch.zeros(num_classes_law, 768))
        nn.init.kaiming_uniform_(self.law_embedding, mode='fan_in')
        self.accu_embedding = nn.Parameter(torch.zeros(num_classes_accu, 768))
        nn.init.kaiming_uniform_(self.accu_embedding, mode='fan_in')
        self.term_embedding = nn.Parameter(torch.zeros(num_classes_term, 768))
        nn.init.kaiming_uniform_(self.term_embedding, mode='fan_in')

        self.segment_embedding = nn.Embedding(3, 768)

        # Different information alignment
        self.case_embedding_transform = nn.Linear(768, 768)

        # Hierarchy
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.model_l1 = nn.TransformerEncoder(transformer_encoder_layer, num_layers=1)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8,batch_first=True)
        self.model_accu = nn.TransformerEncoder(transformer_encoder_layer, num_layers=1)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8,batch_first=True)
        self.model_l3 = nn.TransformerEncoder(transformer_encoder_layer, num_layers=1)

        # DAG mask
        law_accu_edges = np.loadtxt('./datasets/law_accu_edges.txt', dtype='int')
        accu_term_edges = np.loadtxt('./datasets/accu_term_edges.txt', dtype='int')

        # positions with ``True`` is not allowed to attend in Transformer block
        # while ``False`` values will be unchanged.

        dag_mask = 1 - torch.eye(num_classes_law+num_classes_accu)
        for i, j in law_accu_edges:
            dag_mask[i, num_classes_law+j] = 0
        self.dag_mask_l2 = dag_mask.bool().transpose(0, 1)

        dag_mask = 1 - torch.eye(num_classes_accu+num_classes_term)
        for i, j in accu_term_edges:
            dag_mask[i, num_classes_accu+j] = 0
        self.dag_mask_l3 = dag_mask.bool().transpose(0, 1)

        # Predict
        self.linear_l1 = nn.Linear(in_features=768, out_features=1)
        self.linear_accu = nn.Linear(in_features=768, out_features=1)
        self.linear_l3 = nn.Linear(in_features=768, out_features=1)

        # Loss
        self.CE_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()

        if args.froze_bert:
            self.trainable_param_names = ['layer.11', 'layer.10', 'layer.9', 'layer.8', 'layer.7', 'layer.6']
            for name, param in self.model.named_parameters():
                if any(n in name for n in self.trainable_param_names):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, inputs=None, labels_accu=None, labels_law=None, labels_term=None, rt_outputs=False):
        # move data to device
        input_ids = inputs['input_ids'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        case_embedding = outputs['pooler_output'].unsqueeze(1)
        # TODO: Sentence Whitening

        # Transformer
        node_embedding = torch.cat([self.law_embedding, self.accu_embedding, self.term_embedding])
        segment_id = torch.cat([torch.ones(self.num_classes_law).long()*0, torch.ones(self.num_classes_accu).long()*1, torch.ones(self.num_classes_term).long()*2]).to(self.device)
        segment_embedding = self.segment_embedding(segment_id).unsqueeze(0)

        input_embedding = self.case_embedding_transform(case_embedding) + segment_embedding + node_embedding.unsqueeze(0)  # [B, L, 768]

        input_embedding_law, input_embedding_accu, input_embedding_term = self.unpacked_according_to_three_types_of_labels(input_embedding)

        # Three Level
        output_l1 = self.model_l1(input_embedding_law)

        input_accu = torch.cat([output_l1, input_embedding_accu], dim=1)
        output_accu = self.model_accu(input_accu, mask=self.dag_mask_l2.to(self.device))
        _, output_accu = self.unpacked_according_to_law_accu_length(output_accu)

        input_l3 = torch.cat([output_accu, input_embedding_term], dim=1)
        output_l3 = self.model_l3(input_l3, mask=self.dag_mask_l3.to(self.device))
        _, output_term = self.unpacked_according_to_accu_term_length(output_l3)

        # attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
        # source sequence length.
        logits_law_l1 = self.linear_l1(output_l1).squeeze()

        logits_accu_l2 = self.linear_accu(output_accu).squeeze()

        logits_term_l3 = self.linear_l3(output_term).squeeze()

        # Gathering
        logits_law = logits_law_l1
        logits_accu = logits_accu_l2
        logits_term = logits_term_l3

        if rt_outputs:
            input_embedding_law, input_embedding_accu, input_embedding_term = self.unpacked_according_to_three_types_of_labels(segment_embedding + node_embedding.unsqueeze(0))  # [B, L, 768])                     
            mean_cls_embedding = torch.stack([outputs['hidden_states'][i][:, 0, :] for i in range(len(outputs['hidden_states']))], dim=1).mean(1)  # [B, 768]
            #return logits_accu, logits_law, logits_term, input_embedding_law.expand_as(output_l1), input_embedding_accu.expand_as(output_accu), input_embedding_term.expand_as(output_term), mean_cls_embedding
            return logits_accu, logits_law, logits_term

        if self.args.forward_bound:
            # [B, Length]
            probs_law = torch.softmax(logits_law_l1, dim=1)
            probs_accu = torch.softmax(logits_accu_l2, dim=1)
            probs_term = torch.softmax(logits_term_l3, dim=1)

            # [B, Length_accu] = [B, Length_law], [Length_law, Length_accu]
            dag_mask_law_accu = (~self.dag_mask_l2.transpose(0, 1))[:self.num_classes_law, self.num_classes_law:]
            gate_accu = torch.matmul(probs_law, dag_mask_law_accu.to(self.device).float())
            bound_probs_accu = self.normalize_probs(probs_accu * gate_accu)

            # [B, Length_term] = [B, Length_accu], [Length_accu, Length_term]
            dag_mask_accu_term = (~self.dag_mask_l3.transpose(0, 1))[:self.num_classes_accu, self.num_classes_accu:]
            gate_term = torch.matmul(probs_accu, dag_mask_accu_term.to(self.device).float())
            bound_probs_term = self.normalize_probs(probs_term * gate_term)

        if labels_law is not None:
            loss_law = self.CE_loss(logits_law, labels_law)
            loss_accu = self.CE_loss(logits_accu, labels_accu)
            loss_term = self.CE_loss(logits_term, labels_term)

            loss = loss_law + loss_accu + loss_term

            if self.args.forward_bound:
                loss_accu_forword = self.nll_loss(torch.log(bound_probs_accu), labels_accu)
                loss_term_forword = self.nll_loss(torch.log(bound_probs_term), labels_term)

                loss_forward = loss_accu_forword + loss_term_forword
                return loss, loss_forward, logits_accu, logits_law, logits_term, bound_probs_accu, bound_probs_term
            else:
                return loss, logits_accu, logits_law, logits_term

        if self.args.forward_bound:
            return logits_accu, logits_law, logits_term, bound_probs_accu, bound_probs_term
        else:
            return logits_accu, logits_law, logits_term

    def unpacked_according_to_law_accu_length(self, input):
        return input[:, :self.num_classes_law], input[:, self.num_classes_law:]

    def unpacked_according_to_accu_term_length(self, input):
        return input[:, :self.num_classes_accu], input[:, self.num_classes_accu:]

    def unpacked_according_to_three_types_of_labels(self, input):
        return input[:, :self.num_classes_law], input[:, self.num_classes_law:self.num_classes_law+self.num_classes_accu], input[:, self.num_classes_law+self.num_classes_accu:self.num_classes_law+self.num_classes_accu+self.num_classes_term]

    def normalize_probs(self, probs):
        return probs / probs.sum(1, keepdim=True)


class Verifier(nn.Module):
    def __init__(self, device, num_classes_accu=None, num_classes_law=None, num_classes_term=None, args=None):
        super(Verifier, self).__init__()
        self.device = device
        self.args = args

        self.num_classes_accu = num_classes_accu
        self.num_classes_law = num_classes_law
        self.num_classes_term = num_classes_term

        # Label Embedding
        self.node_embedding = nn.Embedding(num_classes_law + num_classes_accu + num_classes_term, 768)

        # DAG mask
        self.model = Classification(device, num_classes_accu=num_classes_accu,
                                    num_classes_law=num_classes_law, num_classes_term=num_classes_term, args=args)

        # 
        self.fact_encoder = AutoModel.from_pretrained('./pre_model/bert-base-chinese')
        self.tokenizer = AutoTokenizer.from_pretrained('./pre_model/bert-base-chinese')

        if args.froze_bert:
            self.trainable_param_names = ['layer.11', 'layer.10', 'layer.9', 'layer.8', 'layer.7', 'layer.6']
            for name, param in self.fact_encoder.named_parameters():
                if any(n in name for n in self.trainable_param_names):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # Relation
        self.segment_embedding = nn.Parameter(torch.randn(3, 768))
        self.linear_transform = nn.Linear(768, 768)
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True)
        self.verification = nn.TransformerEncoder(transformer_encoder_layer, num_layers=3)
        self.linear = nn.Linear(768, 1)

    def forward(self, facts, labels_accu=None, labels_law=None, labels_term=None, is_train=True):
        #B, _, _ = inputs.shape
        # tokenize the data text

        # move data to device
        if labels_accu is not None and labels_law is not None and labels_term is not None:
            labels_accu = torch.from_numpy(np.array(labels_accu)).to(self.device)
            labels_law = torch.from_numpy(np.array(labels_law)).to(self.device)
            labels_term = torch.from_numpy(np.array(labels_term)).to(self.device)
        B = len(facts)
        with torch.no_grad():
            #logits_accu, logits_law, logits_term, output_law, output_accu, output_term, case_embeddings = self.model(inputs, rt_outputs=True)
            logits_accu, logits_law, logits_term = self.model(facts, rt_outputs=True)

        beam_size_law = self.args.beam_size
        beam_size_accu = self.args.beam_size
        beam_size_term = 11
        num_combinations = beam_size_law * beam_size_accu * beam_size_term

        indices_topk_law = torch.topk(logits_law, k=beam_size_law, dim=1).indices.cpu().numpy()
        indices_topk_accu = torch.topk(logits_accu, k=beam_size_accu, dim=1).indices.cpu().numpy() + self.num_classes_law
        indices_topk_term = torch.topk(logits_term, k=beam_size_term, dim=1).indices.cpu().numpy() + self.num_classes_law + self.num_classes_accu
        
        path_batch = []
        if is_train:
            labels_law_np = labels_law.cpu().numpy()
            labels_accu_np = labels_accu.cpu().numpy()
            labels_term_np = labels_term.cpu().numpy()
            path_label_batch = []
        for batch_idx, (indices_topk_law_, indices_topk_accu_, indices_topk_term_) in enumerate(zip(indices_topk_law, indices_topk_accu, indices_topk_term)):
            path_per_sample = []
            path_label_per_sample = []
            for index_law in indices_topk_law_:
                path = []
                path.append(index_law)
                for index_accu in indices_topk_accu_:
                    path = path[:1]
                    path.append(index_accu)
                    for index_term in indices_topk_term_:
                        path = path[:2]
                        path.append(index_term)
                        # path_per_sample.append(torch.cat(path))
                        path_per_sample.append(torch.LongTensor(path))
                        
                        if is_train:
                            if self.args.multi_label:
                                tmp_path_label_per_sample = []
                                if int(labels_law_np[batch_idx]) == int(index_law):
                                    tmp_path_label_per_sample.append(torch.LongTensor([1]))
                                else:
                                    tmp_path_label_per_sample.append(torch.LongTensor([0]))
                                if int(labels_accu_np[batch_idx]) == int(index_accu - self.num_classes_law):
                                    tmp_path_label_per_sample.append(torch.LongTensor([1]))
                                else:
                                    tmp_path_label_per_sample.append(torch.LongTensor([0]))
                                if int(labels_term_np[batch_idx]) == int(index_term - self.num_classes_law - self.num_classes_accu):
                                    tmp_path_label_per_sample.append(torch.LongTensor([1]))
                                else:
                                    tmp_path_label_per_sample.append(torch.LongTensor([0]))
                                path_label_per_sample.append(torch.cat(tmp_path_label_per_sample))
                                
                            else:
                                # TODO: when the gold label can't search by top-k beam search, add the glod triples via gold label (only for training mode). 
                                if int(labels_law_np[batch_idx]) == int(index_law.cpu().numpy()) and \
                                    int(labels_accu_np[batch_idx]) == int(index_accu.cpu().numpy() - self.num_classes_law) and \
                                        int(labels_term_np[batch_idx]) == int(index_term.cpu().numpy() - self.num_classes_law - self.num_classes_accu):
                                    path_label_per_sample.append(torch.LongTensor([1]))
                                else:
                                    path_label_per_sample.append(torch.LongTensor([0]))
        
            path_batch.append(torch.cat(path_per_sample))
            if is_train:
                path_label_batch.append(torch.cat(path_label_per_sample))
    
        #path_batch = torch.cat(path_batch).view(B * num_combinations, 3).to(self.device)  # [b * num_combinations, 3]
        path_batch = torch.cat(path_batch).view(B, num_combinations, 3).to(self.device)  # [b, num_combinations, 3]
    
        if is_train:
            path_label_batch = torch.cat(path_label_batch).to(self.device)   # [b * num_combinations, 1]
        

        # label embedding init
        outputs_select = self.node_embedding(path_batch.data) 

        #outputs = torch.cat([output_law, output_accu, output_term], dim=1).to(self.device) # [b, 233, 768]

        #outputs_select = torch.gather(outputs.repeat(num_combinations), 1, path_batch.unsqueeze(2).expand([path_batch.shape[0], path_batch.shape[1], 768])).squeeze()  # [b * num_combinations, 3, 768]
        
        #outputs_expand = outputs.unsqueeze(1).expand(B, num_combinations, outputs.shape[1], outputs.shape[2])
        #outputs_select = torch.gather(outputs_expand, 2, path_batch.unsqueeze(3).expand([path_batch.shape[0], path_batch.shape[1], path_batch.shape[2], 768])).squeeze()  # [b, num_combinations, 3, 768]
        
        # Get case embeddings
        # segment_facts = []
        # segment_index = [0]
        # for fact in facts:
        #     tmp_segment_facts = fact.split('。')
        #     segment_facts.extend(tmp_segment_facts)
        #     segment_index.append(segment_index[-1] + len(tmp_segment_facts))

        # segment_fact_inputs = tokenizer(list(segment_facts), max_length=256,
        #                     padding=True, truncation=True, return_tensors='pt')

        # segment_fact_inputs['input_ids'] = segment_fact_inputs['input_ids'].to(self.device)
        # segment_fact_inputs['token_type_ids'] = segment_fact_inputs['token_type_ids'].to(self.device)
        # segment_fact_inputs['attention_mask'] = segment_fact_inputs['attention_mask'].to(self.device)
        # with torch.no_grad():
        #     fact_outputs = self.fact_encoder(**segment_fact_inputs)
        
        # # gathering the segment features into a single representation.
        # case_embeddings = []
        # for i in range(len(segment_index)-1):
        #     begin_idx, end_idx = segment_index[i], segment_index[i+1]
        #     tmp_case_embeddings = fact_outputs['pooler_output'][begin_idx: end_idx].max(0).values
        #     case_embeddings.append(tmp_case_embeddings)
        # case_embeddings = torch.stack(case_embeddings)
        # del(fact_outputs)
        inputs = self.tokenizer(list(facts), max_length=512,
                               padding=True, truncation=True, return_tensors='pt')

        inputs['input_ids'] = inputs['input_ids'].to(self.device)
        inputs['token_type_ids'] = inputs['token_type_ids'].to(self.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(self.device)

        case_embeddings = self.fact_encoder(**inputs)['pooler_output']

        case_label_path_inputs = outputs_select + self.linear_transform(case_embeddings).unsqueeze(1).unsqueeze(2).expand(B, num_combinations, 1, 768) + self.segment_embedding.unsqueeze(0).unsqueeze(1)
        # case_label_path_inputs = torch.cat([outputs_select, self.linear_transform(case_embeddings).unsqueeze(1).unsqueeze(2).expand(B, num_combinations, 1, 768)], dim=2)
        case_label_path_inputs = case_label_path_inputs.view(B * num_combinations, 3, 768)
        path_batch = path_batch.view(B * num_combinations, 3)

        # check
        # for i, path_per_sample in enumerate(path_batch):
        #     batch_idx = i // num_combinations
        #     for j in path_per_sample:
        #         assert all(case_label_path_inputs[i, j] == outputs_expand[batch_idx, j])
        
        # for i, path_per_sample in enumerate(path_batch):
        #     batch_idx = i // num_combinations
        #     for type_index, label_index in enumerate(path_per_sample):
        #         assert all(case_label_path_inputs[i, type_index] == outputs[batch_idx, label_index])
        
        if self.args.multi_label:
            verification_outputs = self.linear(self.verification(case_label_path_inputs)).squeeze() # [B * num_combinations, 3]
        else:
            verification_outputs = self.linear(self.verification(case_label_path_inputs).mean(dim=1)) # [B * num_combinations, 1]

        # Return 
        path_batch[:, 1] = path_batch[:, 1] - self.num_classes_law
        path_batch[:, 2] = path_batch[:, 2] - self.num_classes_law - self.num_classes_accu
        
        if not is_train:
            if self.args.multi_label:
                return torch.sigmoid(verification_outputs).view(B, num_combinations, 3), path_batch.view(B, num_combinations, 3)
            else:
                return verification_outputs.view(B, num_combinations), path_batch.view(B, num_combinations, 3)
        else:
            if self.args.multi_label:
                path_label_batch = path_label_batch.view(B * num_combinations, 3)
                if self.args.mse_loss:
                    loss = F.mse_loss(torch.sigmoid(verification_outputs), path_label_batch.float())
                else:
                    loss = F.binary_cross_entropy_with_logits(verification_outputs, path_label_batch.float())

                # path_label_batch_positive = path_label_batch.mean(1)
                # path_label_batch_positive_new = torch.zeros_like(path_label_batch_positive)
                # path_label_batch_positive_new[path_label_batch_positive==1] = 1  # B * num_combinations, 1
                # assert path_label_batch_positive_new.sum().detach().cpu().items() == B,  path_label_batch_positive_new.sum().detach().cpu().items()
                return loss, torch.sigmoid(verification_outputs).view(B, num_combinations, 3), path_batch.view(B, num_combinations, 3)
            else:
                path_label_batch = path_label_batch.view(B * num_combinations, 1)
                # loss = F.binary_cross_entropy_with_logits(verification_outputs, path_label_batch, weight=path_label_batch*(num_combinations-1)+1)
                loss = F.binary_cross_entropy_with_logits(verification_outputs, path_label_batch.float())
                return loss, verification_outputs.view(B, num_combinations), path_batch.view(B, num_combinations, 3)
        
        
