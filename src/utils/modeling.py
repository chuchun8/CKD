import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, BertModel
from utils import modeling, model_utils, model_calib


# BERT
class bert_classifier(nn.Module):

    def __init__(self, num_labels, model_select, dropout):

        super(bert_classifier, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()        
        self.bert = BertModel.from_pretrained("bert-base-uncased", local_files_only=True)
        self.bert.pooler = None
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)
        
    def forward(self, **kwargs):
        
        x_input_ids, x_atten_masks, x_seg_ids = kwargs['input_ids'], kwargs['attention_mask'], kwargs['token_type_ids']      
        last_hidden = self.bert(input_ids=x_input_ids, attention_mask=x_atten_masks, token_type_ids=x_seg_ids)
        cls_hidden = last_hidden[0][:,0]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)
        
        return out
    
# BERTweet
class roberta_large_classifier(nn.Module):

    def __init__(self, num_labels, model_select, dropout):

        super(roberta_large_classifier, self).__init__()

        self.config = AutoConfig.from_pretrained('vinai/bertweet-base', local_files_only=True)
        self.roberta = AutoModel.from_pretrained('vinai/bertweet-base', local_files_only=True)
        self.roberta.pooler = None
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.config.hidden_size, self.config.hidden_size)
        self.out = nn.Linear(self.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        
        # CLS token
        x_input_ids, x_atten_masks = kwargs['input_ids'], kwargs['attention_mask']        
        last_hidden = self.roberta(input_ids=x_input_ids, attention_mask=x_atten_masks)
        cls_hidden = last_hidden[0][:,0]
        query = self.dropout(cls_hidden)
        linear = self.relu(self.linear(query))
        out = self.out(linear)

        return out


# born again networks (ban)
class ban_updater(object):
    
    def __init__(self, **kwargs):
        
        self.model = kwargs.pop("model")
        self.optimizer = kwargs.pop("optimizer")
        self.n_gen = kwargs.pop("n_gen")
        self.s_gen = kwargs.pop("s_gen")
        self.last_model = None

    def update(self, inputs, criterion, percent, T, args):
        
        self.optimizer.zero_grad()
        outputs = self.model(**inputs)
        if self.s_gen > 0:
            self.last_model.eval()
            with torch.no_grad():
                teacher_outputs = self.last_model(**inputs).detach()
            loss = self.kd_loss(outputs, inputs['gt_label'], teacher_outputs, percent, T)
        else:
            loss = criterion(outputs, inputs['gt_label'])
            
        loss.backward()
        if args['clipgradient']:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()
        
        return loss.item()

    def register_last_model(self, weight, num_labels, model_select, device, dropout):
        
        if model_select == 'Bert':
            self.last_model = modeling.bert_classifier(num_labels, model_select, dropout).to(device)
        elif model_select == 'Bertweet':
            self.last_model = modeling.roberta_large_classifier(num_labels, model_select, dropout).to(device)
        self.last_model.load_state_dict(torch.load(weight))
        
    def get_calib_temp(self, valloader, y_val, device, criterion, dataset):
        
        with torch.no_grad():
            preds, _ = model_utils.model_preds(valloader, self.last_model, device, criterion)
            T = model_calib.get_best_temp(preds, y_val, dataset)
        
        return T

    def kd_loss(self, outputs, labels, teacher_outputs, percent, T=1):
            
        KD_loss = T*T*nn.KLDivLoss(reduction='sum')(F.log_softmax(outputs/T,dim=1),F.softmax(teacher_outputs/T,dim=1)) * \
                    (1.-percent) + nn.CrossEntropyLoss(reduction='sum')(outputs, labels) * percent

        return KD_loss
        