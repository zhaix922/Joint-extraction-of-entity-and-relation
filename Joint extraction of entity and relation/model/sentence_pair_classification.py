from typing import *
import numpy as np

import torch
import torch.nn as nn
from transformers import BertModel,BertForPreTraining
import torch.nn.functional as F
from transformers.modeling_outputs import TokenClassifierOutput

from .graph import GLCN



class BERTGLCN(nn.Module):

    def __init__(
            self, 
            pretrained_model_name_or_path,
            dropout_prob,
            num_labels,
            id2label):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        self.glcn = GLCN(
            in_dim=768,
            out_dim=768,
            eta=1,
            gamma=1,
            learning_dim=128,
            num_relation_type = 5,
            num_layers=2
        )
        self.head_sentence_parameter = nn.Parameter(torch.empty(768, 768))
        self.tail_sentence_parameter = nn.Parameter(torch.empty(768, 768))
        self.bias_h = nn.Parameter(torch.empty(768))
        self.classifier_dropout = nn.Dropout(dropout_prob)
        self.num_labels = num_labels
        self.id2label = id2label
        self.classifier = nn.Linear(768, self.num_labels)

        
    @staticmethod
    def compute_mask(mask: torch.Tensor):
        '''
        :param mask: (B, N, T)
        :return: True for masked key position according to pytorch official implementation of Transformer
        '''
        B, N, T = mask.shape
        mask = mask.reshape(B * N, T)
        mask_sum = mask.sum(dim=-1)  # (B*N,)

        # (B*N,)
        graph_node_mask = mask_sum != 0
        # (B * N, T)
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(B * N, T)  # True for valid node
        # If src key are all be masked (indicting text segments is null), atten_weight will be nan after softmax
        # in self-attention layer of Transformer.
        # So we do not mask all padded sample. Instead we mask it after Transformer encoding.
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask  # True for padding mask position
        return src_key_padding_mask, graph_node_mask

    def forward(self, **kwargs):
        # demo:just suport batch-size = 1
        input_ids = kwargs["training_input_ids"]
        attention_mask = kwargs["training_attention_mask"]
        token_type_ids = kwargs["training_token_type_ids"]
        sentence_embeding_idx = kwargs["sentence_embeding_idx"]
        sentence_pair_relation = kwargs["sentence_pair_relation"]
        labels = kwargs["labels"]
        embedding = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids
            )
        text_embedding = embedding[0]
        batch_size,doc_len = sentence_embeding_idx.shape
        batch_sentence_embedding = []
        for batch_idx in range(batch_size):
            sentence_embedding = []
            for sentence_idx in sentence_embeding_idx[batch_idx]:
                sentence_embedding.append(text_embedding[batch_idx][sentence_idx])
            batch_sentence_embedding.append(torch.stack(sentence_embedding))
        x_gcn = torch.stack(batch_sentence_embedding)
        
        # mask for suport batch_size > 1
        # src_key_padding_mask, graph_node_mask = self.compute_mask()
        init_adj = torch.ones((batch_size, doc_len, doc_len),device = "cuda")
        
        # demo:just suport batch_size = 1
        sentence_num = torch.tensor([[doc_len]],device = "cuda") 

        x_gcn, soft_adj, gl_loss = self.glcn(x_gcn, sentence_pair_relation, init_adj,sentence_num)
        adj = soft_adj * init_adj

        ##### Forward Begin #####
        ### Encoder module ###
        # word embedding
        head_sentence_embedding_i = x_gcn.unsqueeze(2).expand(batch_size,doc_len,doc_len,768)
        tail_sentence_embedding_j = x_gcn.unsqueeze(1).expand(batch_size,doc_len,doc_len,768)

        head_sentence_embedding_i = torch.einsum('bijd, dk->bijk', head_sentence_embedding_i, self.head_sentence_parameter)
        tail_sentence_embedding_j = torch.einsum('bijd, dk->bijk', tail_sentence_embedding_j, self.tail_sentence_parameter)

        sentence_pair_embedding = F.leaky_relu(head_sentence_embedding_i + tail_sentence_embedding_j + self.bias_h)
        sentence_pair_output = self.classifier_dropout(sentence_pair_embedding)
        logits = self.classifier(sentence_pair_output)
        logits = logits.view(-1,self.num_labels)
        labels = labels.view(-1).long()
        predict_prob = F.softmax(logits)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(predict_prob, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=predict_prob
        )


    def __str__(self):
        '''
        Model prints with number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
