import torch
import torch.nn as nn
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
from parser import DFG_csharp
from parser import (remove_comments_and_docstrings,
                   tree_to_token_index,
                   index_to_code_token,
                   tree_to_variable_index)
from tree_sitter import Language, Parser
import numpy as np
import os


dfg_function={
    'c':DFG_csharp
}

lang = 'c'
parsers={}        

LANGUAGE = Language('parser/my-languages.so', 'c_sharp')
parser = Parser()
parser.set_language(LANGUAGE) 
parser = [parser,dfg_function[lang]]    
parsers[lang]= parser


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
             input_tokens_1,
             input_ids_1,
             position_idx_1,
             dfg_to_code_1,
             dfg_to_dfg_1,
             pragma_label, private_label, reduction_label

    ):
        #The first code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1=position_idx_1
        self.dfg_to_code_1=dfg_to_code_1
        self.dfg_to_dfg_1=dfg_to_dfg_1
        
        #label
        self.pragma_label=pragma_label
        self.private_label=private_label
        self.reduction_label=reduction_label


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1))
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
    
        
    def forward(self, inputs_ids_1,position_idx_1,attn_mask_1,pragma_labels=None,private_labels=None,reduction_labels=None): 
        bs,l=inputs_ids_1.size()
        inputs_ids=torch.cat((inputs_ids_1.unsqueeze(1),),1).view(bs,l)
        position_idx=torch.cat((position_idx_1.unsqueeze(1),),1).view(bs,l)
        attn_mask=torch.cat((attn_mask_1.unsqueeze(1),),1).view(bs,l,l)

        #embedding
        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.roberta.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx,token_type_ids=position_idx.eq(-1).long())[0]
        logits=self.classifier(outputs)
        # shape: [batch_size, num_classes]
        # prob=F.softmax(logits, dim=-1)
        
        prob=torch.sigmoid(logits)
        if all([label is not None for label in [pragma_labels,private_labels,reduction_labels]]):
            # loss_fct = CrossEntropyLoss()
            labels = torch.stack((pragma_labels,private_labels,reduction_labels), dim=1)
            labels = torch.tensor(labels, dtype=torch.float)
            loss_fct = BCELoss()
            loss = loss_fct(prob, labels)
            return loss,prob
        else:
            return prob
      


class OMPify:

    def __init__(self, model_path, device):
        base_model = 'microsoft/graphcodebert-base'
        self.code_length = 512
        self.data_flow_length = 128
        
        self.device = device
        self.config = RobertaConfig.from_pretrained(base_model)
        self.config.num_labels=3

        self.tokenizer = RobertaTokenizer.from_pretrained(base_model)
    
        model = RobertaForSequenceClassification.from_pretrained(base_model, config=self.config)    
        self.model=Model(model,self.config,self.tokenizer)
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'model.bin')))
        self.model.eval()
        self.model.to(device)

    def extract_dataflow(self, code, parser,lang):
        #remove comments
        try:
            code=remove_comments_and_docstrings(code,lang)
        except:
            pass    
        #obtain dataflow
        if lang=="php":
            code="<?php"+code+"?>"    
        try:
            tree = parser[0].parse(bytes(code,'utf8'))    
            root_node = tree.root_node  
            tokens_index=tree_to_token_index(root_node)     
            code=code.split('\n')
            code_tokens=[index_to_code_token(x,code) for x in tokens_index]  
            index_to_code={}
            for idx,(index,code) in enumerate(zip(tokens_index,code_tokens)):
                index_to_code[index]=(idx,code)  
            try:
                DFG,_=parser[1](root_node,index_to_code,{}) 
            except:
                DFG=[]
            DFG=sorted(DFG,key=lambda x:x[1])
            indexs=set()
            for d in DFG:
                if len(d[-1])!=0:
                    indexs.add(d[1])
                for x in d[-1]:
                    indexs.add(x)
            new_DFG=[]
            for d in DFG:
                if d[1] in indexs:
                    new_DFG.append(d)
            dfg=new_DFG
        except:
            dfg=[]
        return code_tokens,dfg

    def convert_examples_to_features(self, code):
        parser=parsers['c']

        #extract data flow
        code_tokens,dfg=self.extract_dataflow(code,parser,'c')
        code_tokens=[self.tokenizer.tokenize('@ '+x)[1:] if idx!=0 else self.tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
        ori2cur_pos={}
        ori2cur_pos[-1]=(0,0)
        for i in range(len(code_tokens)):
            ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
        code_tokens=[y for x in code_tokens for y in x]  
        
        #truncating
        code_tokens=code_tokens[:self.code_length+self.data_flow_length-3-min(len(dfg),self.data_flow_length)][:512-3]
        source_tokens =[self.tokenizer.cls_token]+code_tokens+[self.tokenizer.sep_token]
        source_ids =  self.tokenizer.convert_tokens_to_ids(source_tokens)
        position_idx = [i+self.tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
        dfg=dfg[:self.code_length+self.data_flow_length-len(source_tokens)]
        source_tokens+=[x[0] for x in dfg]
        position_idx+=[0 for x in dfg]
        source_ids+=[self.tokenizer.unk_token_id for x in dfg]
        padding_length=self.code_length+self.data_flow_length-len(source_ids)
        position_idx+=[self.tokenizer.pad_token_id]*padding_length
        source_ids+=[self.tokenizer.pad_token_id]*padding_length      
        
        #reindex
        reverse_index={}
        for idx,x in enumerate(dfg):
            reverse_index[x[1]]=idx
        for idx,x in enumerate(dfg):
            dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
        dfg_to_dfg=[x[-1] for x in dfg]
        dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
        length=len([self.tokenizer.cls_token])
        dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        

        return InputFeatures(source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg, 0, 0, 0)

    def convert_code(self, code):
        model_input = self.convert_examples_to_features(code)

        #calculate graph-guided masked function
        attn_mask_1= np.zeros((self.code_length+self.data_flow_length,
                        self.code_length+self.data_flow_length),dtype=bool)

        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in model_input.position_idx_1])
        max_length=sum([i!=1 for i in model_input.position_idx_1])
        #sequence can attend to sequence
        attn_mask_1[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(model_input.input_ids_1):
            if i in [0,2]:
                attn_mask_1[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(model_input.dfg_to_code_1):
            if a<node_index and b<node_index:
                attn_mask_1[idx+node_index,a:b]=True
                attn_mask_1[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(model_input.dfg_to_dfg_1):
            for a in nodes:
                if a+node_index<len(model_input.position_idx_1):
                    attn_mask_1[idx+node_index,a+node_index]=True  

        return (torch.tensor(model_input.input_ids_1),
                torch.tensor(model_input.position_idx_1),
                torch.tensor(attn_mask_1),                 
                torch.tensor(model_input.pragma_label),
                torch.tensor(model_input.private_label),
                torch.tensor(model_input.reduction_label))

    def predict(self, loop):
        inputs_ids, position_idx, attn_mask, _, _, _ = self.convert_code(loop)
        logit = self.model(inputs_ids.unsqueeze(dim=0).to(self.device), 
                                    position_idx.unsqueeze(dim=0).to(self.device), 
                                    attn_mask.unsqueeze(dim=0).to(self.device))

        y_pred = logit > 0.5
        pragma, private, reduction = y_pred.squeeze().cpu()

        return bool(pragma), bool(private), bool(reduction)