import torch
import torch.nn as nn
import numpy as np
import math


# Positional encoding (as per "Attention is all you need")
def positional_embeddings(seq_len, d):
    result = torch.ones(seq_len, d)
    for i in range(seq_len):
        for j in range(d):
            result[i][j] = np.sin(i/(10000**(j/d))) if j%2 == 0 else np.cos(i/(10000**((j-1)/d))) 
    return result

#######################################################################################################################
# 1. MulT: MultiModal Transformer ("Multimodal transformer for unaligned multimodal language sequences.")
#######################################################################################################################

## MM encoder block of the MMTransformer
class MMEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # hidden dimension in feedforward network
        self.dropout = dropout
        self.attention_dropout = nn.Dropout(self.dropout)
        self.output_dropout = nn.Dropout(self.dropout)
        
        # layer normalization
        self.lnorm1 = nn.LayerNorm(self.d_model)
        self.lnorm2 = nn.LayerNorm(self.d_model)
        
        # multi-head attention:
        self.ma = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads , dropout=self.dropout, batch_first=True)
        
        # FF layer
        self.ff1 = nn.Sequential(nn.Linear(self.d_model, self.d_ff),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_ff, self.d_model))
        
    def forward(self, source, target):
        
        # computing cross modal attention with dropout and residual connection
        if type(source)!=list:
            padding_mask = None
            target_cma = self.ma(query = self.lnorm1(target), key = self.lnorm1(source), value = self.lnorm1(source), key_padding_mask = padding_mask)[0]
            target_out = target + self.attention_dropout(target_cma)
        else:
            source_txt = source[0]
            padding_mask = source[1]
            target_cma = self.ma(query = self.lnorm1(target), key = self.lnorm1(source_txt), value = self.lnorm1(source_txt), key_padding_mask = padding_mask)[0]
            target_out = target + self.attention_dropout(target_cma)
        # layer norm, feed forward network, and residual connection
        target_out = target_out + self.output_dropout(self.ff1(self.lnorm2(target_out)))
        return target_out

    

class MMTranformer(nn.Module):
    def __init__(self, d_model, max_len, vocab_size,
                 cat_vocab_sizes, num_cat_var,
                 num_numerical_var,
                 n_heads, d_ff, n_layers, 
                 dropout, d_fc, n_classes):
        # super constructor
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size # vocabulary size
        self.max_len = max_len # text sequence length
        self.cat_vocab_sizes = cat_vocab_sizes # list of vocabulary sizes for categorical variables
        self.num_cat_var = num_cat_var # number of categorical variables
        self.num_numerical_var = num_numerical_var # number of numerical variables
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # dimension of the feedforward network model 
        self.n_layers = n_layers # number of encoder layers
        self.dropout = dropout # dropout rate
        self.embedding_dropout = nn.Dropout(dropout) # dropout after text embedding
        self.cat_dropout = nn.Dropout(dropout) # dropout after cat embedding
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.n_classes = n_classes # number of classes
        
        # text embedding, note that with padding = 0: entries do not contribute to the gradient
        self.text_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx = 0) 
           
        # categorical embeddings
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.cat_vocab_sizes[i], embedding_dim=self.d_model, 
                                            padding_idx = 0) for i in range(self.num_cat_var)])
    
        # linear mapper for numerical data
        self.num_linears = nn.ModuleList([nn.Linear(1, self.d_model) for i in range(self.num_numerical_var)])
        
        # classification token [CLS], is learnable
        self.text_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        self.tab_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        
        # positional embedding, not learnable
        self.text_pos_embed = nn.Parameter(data = positional_embeddings(self.max_len + 1, self.d_model), requires_grad = False)
         
        # MM encoder block
        self.MMEncoder_tab_txt = nn.ModuleList([MMEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)])
        self.MMEncoder_txt_tab = nn.ModuleList([MMEncoderLayer(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)])
          
        # Self Attention Transformer encoder
        self.text_encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.n_heads, 
                                                              self.d_ff, dropout=self.dropout, batch_first=True)
        self.text_transformer_encoder = nn.TransformerEncoder(self.text_encoder_layers, self.n_layers)
        self.tab_encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.n_heads, 
                                                              self.d_ff, dropout=self.dropout, batch_first=True)
        self.tab_transformer_encoder = nn.TransformerEncoder(self.tab_encoder_layers, self.n_layers)
        
        
        # last fully connected network
        self.fc1 = nn.Sequential(nn.Linear(2*self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        # embeddings
        nn.init.kaiming_uniform_(self.text_embedding.weight)
        for i in range(self.num_cat_var):
            nn.init.kaiming_uniform_(self.cat_embeddings[i].weight)
        # numerical linear
        for i in range(self.num_numerical_var):
            nn.init.zeros_(self.num_linears[i].bias)
            nn.init.kaiming_uniform_(self.num_linears[i].weight)
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)
            
    def forward(self, texts, padding_mask, categoricals, numericals):
        # 1. reshape categoricals for embeddings and numericals before linear transformation 
        categorical_list = [categoricals[:,i].unsqueeze(dim=1) for i in range(self.num_cat_var)]
        numerical_list = [numericals[:,i].unsqueeze(dim=1).unsqueeze(dim=1) for i in range(self.num_numerical_var)]
        
        # 2. embedding layers
        texts = self.text_embedding(texts) 
        texts = self.embedding_dropout(texts)
        cat_embedding_list = [self.cat_embeddings[i](categorical_list[i]) for i in range(self.num_cat_var)]
        categoricals = torch.cat([cat_embedding_list[i] for i in range(self.num_cat_var)], dim = 1)
        categoricals = self.cat_dropout(categoricals)
        numerical_embedding_list = [self.num_linears[i](numerical_list[i].float()) for i in range(self.num_numerical_var)]
        numericals = torch.cat([numerical_embedding_list[i] for i in range(self.num_numerical_var)], dim = 1)
        tabulars = torch.cat([categoricals, numericals], dim = 1) # concatenate categorical and numerical embeddings
        
        # 3. classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        texts = torch.stack([torch.vstack((self.text_cls, texts[i])) for i in range(len(texts))]) * math.sqrt(self.d_model)
        tabulars = torch.stack([torch.vstack((self.tab_cls, tabulars[i])) for i in range(len(tabulars))]) * math.sqrt(self.d_model)
        
        # 4. positional embeddings
        text_pos_embeds = self.text_pos_embed.repeat(len(texts),1,1)
        texts = texts + text_pos_embeds
        
        ## 5. MM encoder
        texts_dict = {}
        texts_dict[0] = texts
        tabulars_dict = {}
        tabulars_dict[0] = tabulars
        for i, layer in enumerate(self.MMEncoder_tab_txt):
            texts_dict[i+1] = layer(source = tabulars_dict[0], target = texts_dict[i])
        for i, layer in enumerate(self.MMEncoder_txt_tab):
            tabulars_dict[i+1] = layer(source = [texts_dict[0], padding_mask], target = tabulars_dict[i])
        texts = texts_dict[i+1]
        tabulars = tabulars_dict[i+1]
        
        # 6. Self attention Transformer encoder
        texts = self.text_transformer_encoder(texts, src_key_padding_mask = padding_mask)
        tabulars = self.tab_transformer_encoder(tabulars)
        
        # 7. Concatenate CLS tokens
        text_cls = texts[:,0,:]
        tabular_cls = tabulars[:,0,:]
        mm_cls = torch.cat([text_cls, tabular_cls], dim = 1)

        # 8. Fully connected network for classification purpose
        pred = self.fc1(mm_cls)
        
        return pred
    
#######################################################################################################################
# 2. EarlyConcat: Early concatenation Transformer (as does for example VideoBERT for text and video) 
#######################################################################################################################

class EarlyConcatTranformer(nn.Module):
    def __init__(self, d_model, max_len, vocab_size,
                 cat_vocab_sizes, num_cat_var,
                 num_numerical_var,
                 n_heads, d_ff, n_layers, 
                 dropout, d_fc, n_classes):
        # super constructor
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size # vocabulary size
        self.max_len = max_len # text sequence length
        self.cat_vocab_sizes = cat_vocab_sizes # list of vocabulary sizes for categorical variables
        self.num_cat_var = num_cat_var # number of categorical variables
        self.num_numerical_var = num_numerical_var # number of numerical variables
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # dimension of the feedforward network model 
        self.n_layers = n_layers # number of encoder layers
        self.dropout = dropout # dropout rate
        self.embedding_dropout = nn.Dropout(dropout) # dropout after text embedding
        self.cat_dropout = nn.Dropout(dropout) # dropout after cat embedding
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.n_classes = n_classes # number of classes
        
        # text embedding, note that with padding = 0: entries do not contribute to the gradient
        self.text_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx = 0) 
           
        # categorical embeddings
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.cat_vocab_sizes[i], embedding_dim=self.d_model, 
                                            padding_idx = 0) for i in range(self.num_cat_var)])
    
        # linear mapper for tabular data
        self.num_linears = nn.ModuleList([nn.Linear(1, self.d_model) for i in range(self.num_numerical_var)])
        
        # classification token [CLS], is learnable
        self.concats_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        
        # positional embedding, not learnable
        self.num_var = self.num_cat_var + self.num_numerical_var + self.max_len
        self.pos_embed = nn.Parameter(data = positional_embeddings(self.num_var + 1, self.d_model), requires_grad = False)
         
        # Self Attention Transformer encoder
        self.encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.n_heads, 
                                                              self.d_ff, dropout=self.dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, self.n_layers)
        
        # last fully connected network
        self.fc1 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        # embeddings
        nn.init.kaiming_uniform_(self.text_embedding.weight)
        for i in range(self.num_cat_var):
            nn.init.kaiming_uniform_(self.cat_embeddings[i].weight)
        # numerical linear
        for i in range(self.num_numerical_var):
            nn.init.zeros_(self.num_linears[i].bias)
            nn.init.kaiming_uniform_(self.num_linears[i].weight)
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)
            
    def forward(self, texts, padding_mask, categoricals, numericals):
 
        # 1. reshape categoricals for embeddings and numericals before linear transformation 
        categorical_list = [categoricals[:,i].unsqueeze(dim=1) for i in range(self.num_cat_var)]
        numerical_list = [numericals[:,i].unsqueeze(dim=1).unsqueeze(dim=1) for i in range(self.num_numerical_var)]
        
        # 2. embedding layers
        texts = self.text_embedding(texts) 
        texts = self.embedding_dropout(texts)
        cat_embedding_list = [self.cat_embeddings[i](categorical_list[i]) for i in range(self.num_cat_var)]
        categoricals = torch.cat([cat_embedding_list[i] for i in range(self.num_cat_var)], dim = 1)
        categoricals = self.cat_dropout(categoricals)
        numerical_embedding_list = [self.num_linears[i](numerical_list[i].float()) for i in range(self.num_numerical_var)]
        numericals = torch.cat([numerical_embedding_list[i] for i in range(self.num_numerical_var)], dim = 1)
        
        # 3. concatenate modalities
        concats = torch.cat([categoricals, numericals, texts], dim = 1) # concatenate all modalities
        
        # 4. classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        concats = torch.stack([torch.vstack((self.concats_cls, concats[i])) for i in range(len(concats))]) * math.sqrt(self.d_model)

        # 5. positional embeddings
        pos_embeds = self.pos_embed.repeat(len(concats),1,1)
        concats = concats + pos_embeds
       
        # 6. Self attention Transformer encoder        
        concats = self.transformer_encoder(concats, src_key_padding_mask = padding_mask)
        
        # 7. Predict with CLS tokens
        concats_cls = concats[:,0,:]
        pred = self.fc1(concats_cls)
        
        return pred
    
#######################################################################################################################
# 3. LateFuse ("Multimodal-Toolkit: A Package for Learning on Tabular and Text Data with Transformers")
#######################################################################################################################
   
class TabularTranformer(nn.Module):
    def __init__(self, d_model, max_len, vocab_size,
                 num_cat_var,
                 num_numerical_var,
                 n_heads, d_ff, n_layers, 
                 dropout, d_fc, n_classes):
        # super constructor
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size # vocabulary size
        self.max_len = max_len # text sequence length
        self.num_cat_var = num_cat_var # number of categorical variables
        self.num_numerical_var = num_numerical_var # number of numerical variables
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # dimension of the feedforward network model 
        self.n_layers = n_layers # number of encoder layers
        self.dropout = dropout # dropout rate
        self.embedding_dropout = nn.Dropout(dropout) # dropout after text embedding
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.n_classes = n_classes # number of classes
        
        # text embedding, note that with padding = 0: entries do not contribute to the gradient
        self.text_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx = 0) 
        
        # classification token [CLS], is learnable
        self.text_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        
        # positional embedding, not learnable
        self.text_pos_embed = nn.Parameter(data = positional_embeddings(self.max_len + 1, self.d_model), requires_grad = False)
         
        # Self Attention Transformer encoder
        self.text_encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.n_heads, 
                                                              self.d_ff, dropout=self.dropout, batch_first=True)
        self.text_transformer_encoder = nn.TransformerEncoder(self.text_encoder_layers, self.n_layers)
        
        # last fully connected network
        self.fc1 = nn.Sequential(nn.Linear(self.d_model+self.num_cat_var+self.num_numerical_var, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        # embeddings
        nn.init.kaiming_uniform_(self.text_embedding.weight)
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)
            
    def forward(self, texts, padding_mask, categoricals, numericals):

        # 1. concatenate categoricals and numericals 
        tabulars = torch.cat([categoricals.float(), numericals.float()], dim = 1)
        
        # 2. embedding layers
        texts = self.text_embedding(texts) 
        texts = self.embedding_dropout(texts)
        
        # 3. classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        texts = torch.stack([torch.vstack((self.text_cls, texts[i])) for i in range(len(texts))]) * math.sqrt(self.d_model)
       
        # 4. positional embeddings
        text_pos_embeds = self.text_pos_embed.repeat(len(texts),1,1)
        texts = texts + text_pos_embeds

        # 5. Self attention Transformer encoder
        texts = self.text_transformer_encoder(texts, src_key_padding_mask = padding_mask)
        
        # 7. Concatenate CLS token and tabular data
        text_cls = texts[:,0,:]
        mm_cls = torch.cat([text_cls, tabulars], dim = 1)

        # 8. Fully connected network for classification purpose
        pred = self.fc1(mm_cls)
        
        return pred
    

#######################################################################################################################
# 4. Tensor Fusion Network "Tensor Fusion Network for Multimodal Sentiment Analysis"
#######################################################################################################################

class Tabular_Model(nn.Module):
    """
    Tabular subnet
    """
    def __init__(self, d_model, num_cat_var, num_numerical_var, dropout):
        super(Tabular_Model, self).__init__()
        
        self.input_size = num_cat_var + num_numerical_var
        self.d_model = d_model
        self.dropout= dropout
        
        self.fc1 = nn.Linear(self.input_size,self.d_model)
        self.fc2 = nn.Linear(self.d_model,self.d_model)
        self.dropout = nn.Dropout(self.dropout)
        self.fc3 = nn.Linear(self.d_model,self.d_model)
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.kaiming_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc3.bias)
        nn.init.kaiming_uniform_(self.fc3.weight)

    def forward(self,x):
        return self.fc3(self.dropout(torch.relu(self.fc2(self.dropout(torch.relu(self.fc1(x)))))))
    
class Text_Model(nn.Module):
    """
    Text subnet
    """    
    def __init__(self, d_model, vocab_size, dropout):
        super(Text_Model, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dropout= dropout
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx=0)
        self.lstm = nn.LSTM(input_size =self.d_model, hidden_size=self.d_model, num_layers=1, batch_first=True,
                           bidirectional=False)
        self.dropout = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.d_model,self.d_model)
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        # embeddings
        nn.init.kaiming_uniform_(self.embeddings.weight)
        # final FC network
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc1.weight)
        
    def forward(self,x):
        x = self.embeddings(x)
        x= self.dropout(x)
        output, (h_n, c_n) = self.lstm(x)
        h_n = self.dropout(h_n.squeeze(dim=0))
        z = self.fc1(h_n) 
        return z

class TFN(nn.Module):
    """
    Tensor Fusion Network
    """
    def __init__(self, d_model, vocab_size, num_cat_var, num_numerical_var, dropout, d_fc, n_classes, device):
        super(TFN, self).__init__()
        # attributes
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_cat_var = num_cat_var
        self.num_numerical_var = num_numerical_var
        self.d_fc = d_fc
        self.dropout = dropout
        self.n_classes = n_classes
        self.device = device
        
        # define the pre-fusion subnetworks
        self.tabular_subnet = Tabular_Model(self.d_model, self.num_cat_var, self.num_numerical_var, self.dropout)
        self.text_subnet = Text_Model(self.d_model, self.vocab_size, self.dropout)

        # define the post fusion layers
        self.post_fusion_dropout = nn.Dropout(self.dropout)
        self.post_fusion_layer_1 = nn.Linear((self.d_model + 1) * (self.d_model + 1), self.d_fc)
        self.post_fusion_layer_2 = nn.Linear(self.d_fc, self.n_classes)
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        # final FC network
        nn.init.zeros_(self.post_fusion_layer_1.bias)
        nn.init.kaiming_uniform_(self.post_fusion_layer_1.weight)
        nn.init.zeros_(self.post_fusion_layer_2.bias)
        nn.init.kaiming_uniform_(self.post_fusion_layer_2.weight)

    def forward(self, texts, padding_mask, categoricals, numericals):
        # pre-fusion: outputs of subnetworks
        tab_x = torch.cat([categoricals.float(), numericals.float()] ,dim=1)
        tabular_h = self.tabular_subnet(tab_x)
        text_h = self.text_subnet(texts)
        batch_size = len(texts)
        
        # add 1
        _tabular_h = torch.cat((torch.ones(batch_size, 1, requires_grad=False).to(self.device), tabular_h), dim=1)
        _text_h = torch.cat((torch.ones(batch_size, 1, requires_grad=False).to(self.device), text_h), dim=1)
                      
        # fusion: outer product (batch_size, tabular_h_dim + 1, 1) x (batch_size, text_batch_dim + 1, 1), we flatten the x-D tensor  
        fusion_tensor = torch.bmm(_tabular_h.unsqueeze(2), _text_h.unsqueeze(1)).view(batch_size, -1)
        
        # post-fusion layers
        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)
        post_fusion_y1 = torch.relu(self.post_fusion_layer_1(post_fusion_dropped))
        post_fusion_y2 = self.post_fusion_layer_2(post_fusion_y1)
        
        return post_fusion_y2
    
#######################################################################################################################
# 5. Tabular-Text Transformer or TTT (embeddings for continuous variables, one-versus-all attention, late fusion)
#######################################################################################################################

## MM encoder block of the MMTransformer
class MMEncoderLayer2(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # hidden dimension in feedforward network
        self.dropout = dropout
        self.attention_dropout = nn.Dropout(self.dropout)
        self.output_dropout = nn.Dropout(self.dropout)
        
        # layer normalization
        self.lnorm1 = nn.LayerNorm(self.d_model)
        self.lnorm2 = nn.LayerNorm(self.d_model)
        
        # multi-head attention:
        self.ma = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=self.n_heads , dropout=self.dropout, batch_first=True)
        
        # FF layer
        self.ff1 = nn.Sequential(nn.Linear(self.d_model, self.d_ff),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_ff, self.d_model))
        
    def forward(self, source, target):
        
        # computing one-versus-all attention
        if type(source)!=list:
            target_txt = target[0]
            padding_mask = target[1]
            source = torch.cat([source, target_txt], dim = 1)
            target_cma = self.ma(query = self.lnorm1(target_txt), key = self.lnorm1(source), value = self.lnorm1(source), key_padding_mask = padding_mask)[0]
            target_out = target_txt + self.attention_dropout(target_cma)
        else:
            source_txt = source[0]
            padding_mask = source[1]
            source_txt = torch.cat([target, source_txt], dim = 1)
            target_cma = self.ma(query = self.lnorm1(target), key = self.lnorm1(source_txt), value = self.lnorm1(source_txt), key_padding_mask = padding_mask)[0]
            target_out = target + self.attention_dropout(target_cma)
            
        # layer norm, feed forward network, and residual connection
        target_out = target_out + self.output_dropout(self.ff1(self.lnorm2(target_out)))
        
        return target_out

    

class TTT(nn.Module):
    def __init__(self, d_model, max_len, vocab_size,
                 cat_vocab_sizes, num_cat_var,
                 num_numerical_var, quantiles,
                 n_heads, d_ff, n_layers, 
                 dropout, d_fc, n_classes, device):
        # super constructor
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size # vocabulary size
        self.max_len = max_len # text sequence length
        self.cat_vocab_sizes = cat_vocab_sizes # list of vocabulary sizes for categorical variables
        self.num_cat_var = num_cat_var # number of categorical variables
        self.num_numerical_var = num_numerical_var # number of numerical variables
        self.quantiles = quantiles # quantiles for each numerical variable
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # dimension of the feedforward network model 
        self.n_layers = n_layers # number of encoder layers
        self.dropout = dropout # dropout rate
        self.embedding_dropout = nn.Dropout(dropout) # dropout after text embedding
        self.cat_dropout = nn.Dropout(dropout) # dropout after cat embedding
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.n_classes = n_classes # number of classes
        self.device = device
        
        # text embedding, note that with padding = 0: entries do not contribute to the gradient
        self.text_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx = 0) 
           
        # categorical embeddings
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.cat_vocab_sizes[i], embedding_dim=self.d_model, 
                                            padding_idx = 0) for i in range(self.num_cat_var)])
    
        # embeddings for numericals
        self.num_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=len(self.quantiles[i]), embedding_dim=self.d_model) for i in range(self.num_numerical_var)])
        
        # classification token [CLS], is learnable
        self.text_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        self.tab_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        
        # positional embedding, not learnable
        self.text_pos_embed = nn.Parameter(data = positional_embeddings(self.max_len + 1, self.d_model), requires_grad = False)
         
        # MM encoder block
        self.MMEncoder_tab_txt = nn.ModuleList([MMEncoderLayer2(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)])
        self.MMEncoder_txt_tab = nn.ModuleList([MMEncoderLayer2(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)])
        
        # text fully connected network
        self.fc1 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # tabular fully connected network
        self.fc2 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        # embeddings
        nn.init.kaiming_uniform_(self.text_embedding.weight)
        for i in range(self.num_cat_var):
            nn.init.kaiming_uniform_(self.cat_embeddings[i].weight)
        for i in range(self.num_numerical_var):
            nn.init.kaiming_uniform_(self.num_embeddings[i].weight)
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)
        nn.init.zeros_(self.fc2[0].bias)
        nn.init.kaiming_uniform_(self.fc2[0].weight)
        nn.init.zeros_(self.fc2[3].bias)
        nn.init.kaiming_uniform_(self.fc2[3].weight)
            
    def forward(self, texts, padding_mask, categoricals, numericals):

        # 1. reshape categoricals and numericals for embeddings
        categorical_list = [categoricals[:,i].unsqueeze(dim=1) for i in range(self.num_cat_var)]
        C = torch.zeros((len(numericals), len(self.quantiles[0])), dtype = int).to(self.device)
        for i in range(len(self.quantiles[0])):
            C[:,i] = i
        numerical_token_list = [C for i in range(self.num_numerical_var)]
        weights = [np.abs(np.array([numericals[:,i].cpu().numpy() for j in range(len(self.quantiles[i]))]).T - self.quantiles[i]) for i in range(self.num_numerical_var)] # distance to quantile
        numericals = numericals.to(self.device) 
        weights = [torch.tensor(1/w).to(self.device)  for w in weights] # similarity
        weights = [weights[i]/weights[i].sum(dim=1, keepdim=True) for i in range(self.num_numerical_var)] # normalize
        weights = [torch.nan_to_num(weights[i], 1.) for i in range(self.num_numerical_var)] # replace nan by 1
        weights = [weights[i]/weights[i].sum(dim=1, keepdim=True) for i in range(self.num_numerical_var)] # normalize
        weights = [weights[i].unsqueeze(1) for i in range(self.num_numerical_var)]
        weights = [weights[i].float() for i in range(self.num_numerical_var)]
       
        # 2. embedding layers
        # text embedding
        texts = self.text_embedding(texts) 
        texts = self.embedding_dropout(texts)
        # categorical embedding
        cat_embedding_list = [self.cat_embeddings[i](categorical_list[i]) for i in range(self.num_cat_var)]
        categoricals = torch.cat([cat_embedding_list[i] for i in range(self.num_cat_var)], dim = 1)
        categoricals = self.cat_dropout(categoricals)
        # numerical embedding
        num_embedding_list = [self.num_embeddings[i](numerical_token_list[i]) for i in range(self.num_numerical_var)]
        # numericals: Weights x Quantile Embeddings
        numericals = [torch.bmm(weights[i], num_embedding_list[i]) for i in range(self.num_numerical_var)]
        numericals = torch.cat([numericals[i] for i in range(self.num_numerical_var)], dim=1)
        tabulars = torch.cat([categoricals, numericals], dim = 1) # concatenate categorical and numerical embeddings
        
        # 3. classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        texts = torch.stack([torch.vstack((self.text_cls, texts[i])) for i in range(len(texts))]) * math.sqrt(self.d_model)
        tabulars = torch.stack([torch.vstack((self.tab_cls, tabulars[i])) for i in range(len(tabulars))]) * math.sqrt(self.d_model)
        
        # 4. positional embeddings
        text_pos_embeds = self.text_pos_embed.repeat(len(texts),1,1)
        texts = texts + text_pos_embeds

        ## 5. MM encoder
        texts_dict = {}
        texts_dict[0] = texts
        tabulars_dict = {}
        tabulars_dict[0] = tabulars
        for i, layer in enumerate(self.MMEncoder_tab_txt):
            texts_dict[i+1] = layer(source = tabulars_dict[0], target = [texts_dict[i], padding_mask])
        for i, layer in enumerate(self.MMEncoder_txt_tab):
            tabulars_dict[i+1] = layer(source = [texts_dict[0], padding_mask], target = tabulars_dict[i])
        texts = texts_dict[i+1]
        tabulars = tabulars_dict[i+1]

        # 6. Extract CLS tokens
        text_cls = texts[:,0,:]
        tabular_cls = tabulars[:,0,:]
        
        #7. Late fusion (average of logits)
        text_pred = self.fc1(text_cls)
        tabular_pred = self.fc2(tabular_cls)
        pred = (text_pred + tabular_pred)/2 # average
        
        return pred, text_pred, tabular_pred
        

        
#######################################################################################################################
# 6. Ablation study 1: TTT with linear embedding instead of distance-to-quantile
#######################################################################################################################

class TTT_ablation1(nn.Module):
    def __init__(self, d_model, max_len, vocab_size,
                 cat_vocab_sizes, num_cat_var,
                 num_numerical_var,
                 n_heads, d_ff, n_layers, 
                 dropout, d_fc, n_classes, device):
        # super constructor
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size # vocabulary size
        self.max_len = max_len # text sequence length
        self.cat_vocab_sizes = cat_vocab_sizes # list of vocabulary sizes for categorical variables
        self.num_cat_var = num_cat_var # number of categorical variables
        self.num_numerical_var = num_numerical_var # number of numerical variables
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # dimension of the feedforward network model 
        self.n_layers = n_layers # number of encoder layers
        self.dropout = dropout # dropout rate
        self.embedding_dropout = nn.Dropout(dropout) # dropout after text embedding
        self.cat_dropout = nn.Dropout(dropout) # dropout after cat embedding
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.n_classes = n_classes # number of classes
        self.device = device
        
        # text embedding, note that with padding = 0: entries do not contribute to the gradient
        self.text_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx = 0) 
           
        # categorical embeddings
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.cat_vocab_sizes[i], embedding_dim=self.d_model, 
                                            padding_idx = 0) for i in range(self.num_cat_var)])
    
        
        # linear mapper for numericals
        self.num_linears = nn.ModuleList([nn.Linear(1, self.d_model) for i in range(self.num_numerical_var)])
        
        # classification token [CLS], is learnable
        self.text_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        self.tab_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        
        # positional embedding, not learnable
        self.text_pos_embed = nn.Parameter(data = positional_embeddings(self.max_len + 1, self.d_model), requires_grad = False)
         
        # MM encoder block
        self.MMEncoder_tab_txt = nn.ModuleList([MMEncoderLayer2(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)])
        self.MMEncoder_txt_tab = nn.ModuleList([MMEncoderLayer2(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)])
        
        # text fully connected network
        self.fc1 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # tabular fully connected network
        self.fc2 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        # embeddings
        nn.init.kaiming_uniform_(self.text_embedding.weight)
        for i in range(self.num_cat_var):
            nn.init.kaiming_uniform_(self.cat_embeddings[i].weight)

        # numerical linear
        for i in range(self.num_numerical_var):
            nn.init.zeros_(self.num_linears[i].bias)
            nn.init.kaiming_uniform_(self.num_linears[i].weight)
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)
        nn.init.zeros_(self.fc2[0].bias)
        nn.init.kaiming_uniform_(self.fc2[0].weight)
        nn.init.zeros_(self.fc2[3].bias)
        nn.init.kaiming_uniform_(self.fc2[3].weight)
            
    def forward(self, texts, padding_mask, categoricals, numericals):

        # 1. reshape categoricals and numericals for embeddings
        categorical_list = [categoricals[:,i].unsqueeze(dim=1) for i in range(self.num_cat_var)]
        numerical_list = [numericals[:,i].unsqueeze(dim=1).unsqueeze(dim=1) for i in range(self.num_numerical_var)]
       
        # 2. embedding layers
        # text embedding
        texts = self.text_embedding(texts) 
        texts = self.embedding_dropout(texts)
        # categorical embedding
        cat_embedding_list = [self.cat_embeddings[i](categorical_list[i]) for i in range(self.num_cat_var)]
        categoricals = torch.cat([cat_embedding_list[i] for i in range(self.num_cat_var)], dim = 1)
        categoricals = self.cat_dropout(categoricals)
        # numerical embedding
        numerical_embedding_list = [self.num_linears[i](numerical_list[i].float()) for i in range(self.num_numerical_var)]
        numericals = torch.cat([numerical_embedding_list[i] for i in range(self.num_numerical_var)], dim = 1)
        tabulars = torch.cat([categoricals, numericals], dim = 1) # concatenate categorical and numerical embeddings
        
        # 3. classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        texts = torch.stack([torch.vstack((self.text_cls, texts[i])) for i in range(len(texts))]) * math.sqrt(self.d_model)
        tabulars = torch.stack([torch.vstack((self.tab_cls, tabulars[i])) for i in range(len(tabulars))]) * math.sqrt(self.d_model)
        
        # 4. positional embeddings
        text_pos_embeds = self.text_pos_embed.repeat(len(texts),1,1)
        texts = texts + text_pos_embeds

        ## 5. MM encoder
        texts_dict = {}
        texts_dict[0] = texts
        tabulars_dict = {}
        tabulars_dict[0] = tabulars
        for i, layer in enumerate(self.MMEncoder_tab_txt):
            texts_dict[i+1] = layer(source = tabulars_dict[0], target = [texts_dict[i], padding_mask])
        for i, layer in enumerate(self.MMEncoder_txt_tab):
            tabulars_dict[i+1] = layer(source = [texts_dict[0], padding_mask], target = tabulars_dict[i])
        texts = texts_dict[i+1]
        tabulars = tabulars_dict[i+1]
       
        # 6. Extract CLS tokens
        text_cls = texts[:,0,:]
        tabular_cls = tabulars[:,0,:]
        
        #7. Late fusion (average of logits)
        text_pred = self.fc1(text_cls)
        tabular_pred = self.fc2(tabular_cls)
        pred = (text_pred + tabular_pred)/2 # average
        
        return pred, text_pred, tabular_pred
    
    
#######################################################################################################################
# 7. Ablation study 2: TTT with self-attention instead of overall attention
#######################################################################################################################


class TTT_ablation2(nn.Module):
    def __init__(self, d_model, max_len, vocab_size,
                 cat_vocab_sizes, num_cat_var,
                 num_numerical_var, quantiles,
                 n_heads, d_ff, n_layers, 
                 dropout, d_fc, n_classes, device):
        # super constructor
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size # vocabulary size
        self.max_len = max_len # text sequence length
        self.cat_vocab_sizes = cat_vocab_sizes # list of vocabulary sizes for categorical variables
        self.num_cat_var = num_cat_var # number of categorical variables
        self.num_numerical_var = num_numerical_var # number of numerical variables
        self.quantiles = quantiles # quantiles for each numerical variable
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # dimension of the feedforward network model 
        self.n_layers = n_layers # number of encoder layers
        self.dropout = dropout # dropout rate
        self.embedding_dropout = nn.Dropout(dropout) # dropout after text embedding
        self.cat_dropout = nn.Dropout(dropout) # dropout after cat embedding
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.n_classes = n_classes # number of classes
        self.device = device
        
        # text embedding, note that with padding = 0: entries do not contribute to the gradient
        self.text_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx = 0) 
           
        # categorical embeddings
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.cat_vocab_sizes[i], embedding_dim=self.d_model, 
                                            padding_idx = 0) for i in range(self.num_cat_var)])
    
        # embeddings for numericals
        self.num_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=len(self.quantiles[i]), embedding_dim=self.d_model) for i in range(self.num_numerical_var)])
        
        # classification token [CLS], is learnable
        self.text_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        self.tab_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        
        # positional embedding, not learnable
        self.text_pos_embed = nn.Parameter(data = positional_embeddings(self.max_len + 1, self.d_model), requires_grad = False)
         
        # Self Attention Transformer encoder
        self.text_encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.n_heads, 
                                                              self.d_ff, dropout=self.dropout, batch_first=True)
        self.text_transformer_encoder = nn.TransformerEncoder(self.text_encoder_layers, self.n_layers)
        self.tab_encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.n_heads, 
                                                              self.d_ff, dropout=self.dropout, batch_first=True)
        self.tab_transformer_encoder = nn.TransformerEncoder(self.tab_encoder_layers, self.n_layers)
        
        
        # text fully connected network
        self.fc1 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # tabular fully connected network
        self.fc2 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        # embeddings
        nn.init.kaiming_uniform_(self.text_embedding.weight)
        for i in range(self.num_cat_var):
            nn.init.kaiming_uniform_(self.cat_embeddings[i].weight)
        for i in range(self.num_numerical_var):
            nn.init.kaiming_uniform_(self.num_embeddings[i].weight)
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)
        nn.init.zeros_(self.fc2[0].bias)
        nn.init.kaiming_uniform_(self.fc2[0].weight)
        nn.init.zeros_(self.fc2[3].bias)
        nn.init.kaiming_uniform_(self.fc2[3].weight)
            
    def forward(self, texts, padding_mask, categoricals, numericals):

        # 1. reshape categoricals and numericals for embeddings
        categorical_list = [categoricals[:,i].unsqueeze(dim=1) for i in range(self.num_cat_var)]
        C = torch.zeros((len(numericals), len(self.quantiles[0])), dtype = int).to(self.device)
        for i in range(len(self.quantiles[0])):
            C[:,i] = i
        numerical_token_list = [C for i in range(self.num_numerical_var)]
        weights = [np.abs(np.array([numericals[:,i].cpu().numpy() for j in range(len(self.quantiles[i]))]).T - self.quantiles[i]) for i in range(self.num_numerical_var)] # distance to quantile
        numericals = numericals.to(self.device) 
        weights = [torch.tensor(1/w).to(self.device)  for w in weights] # similarity
        weights = [weights[i]/weights[i].sum(dim=1, keepdim=True) for i in range(self.num_numerical_var)] # normalize
        weights = [torch.nan_to_num(weights[i], 1.) for i in range(self.num_numerical_var)] # replace nan by 1
        weights = [weights[i]/weights[i].sum(dim=1, keepdim=True) for i in range(self.num_numerical_var)] # normalize
        weights = [weights[i].unsqueeze(1) for i in range(self.num_numerical_var)]
        weights = [weights[i].float() for i in range(self.num_numerical_var)]
       
        # 2. embedding layers
        # text embedding
        texts = self.text_embedding(texts) 
        texts = self.embedding_dropout(texts)
        # categorical embedding
        cat_embedding_list = [self.cat_embeddings[i](categorical_list[i]) for i in range(self.num_cat_var)]
        categoricals = torch.cat([cat_embedding_list[i] for i in range(self.num_cat_var)], dim = 1)
        categoricals = self.cat_dropout(categoricals)
        # numerical embedding
        num_embedding_list = [self.num_embeddings[i](numerical_token_list[i]) for i in range(self.num_numerical_var)]
        # numericals: Weights x Quantile Embeddings
        numericals = [torch.bmm(weights[i], num_embedding_list[i]) for i in range(self.num_numerical_var)]
        numericals = torch.cat([numericals[i] for i in range(self.num_numerical_var)], dim=1)
        tabulars = torch.cat([categoricals, numericals], dim = 1) # concatenate categorical and numerical embeddings
        
        # 3. classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        texts = torch.stack([torch.vstack((self.text_cls, texts[i])) for i in range(len(texts))]) * math.sqrt(self.d_model)
        tabulars = torch.stack([torch.vstack((self.tab_cls, tabulars[i])) for i in range(len(tabulars))]) * math.sqrt(self.d_model)
        
        # 4. positional embeddings
        text_pos_embeds = self.text_pos_embed.repeat(len(texts),1,1)
        texts = texts + text_pos_embeds

        # 5. Self attention Transformer encoder
        texts = self.text_transformer_encoder(texts, src_key_padding_mask = padding_mask)
        tabulars = self.tab_transformer_encoder(tabulars)
       
        # 6. Extract CLS tokens
        text_cls = texts[:,0,:]
        tabular_cls = tabulars[:,0,:]
        
        # 7. Late fusion (average of logits)
        text_pred = self.fc1(text_cls)
        tabular_pred = self.fc2(tabular_cls)
        pred = (text_pred + tabular_pred)/2 # average
        
        return pred, text_pred, tabular_pred
    
    

#######################################################################################################################
# 8. Ablation study 3: TTT with 1 loss based on the average of logits instead of one loss by modality stream
#######################################################################################################################

class TTT_ablation3(nn.Module):
    def __init__(self, d_model, max_len, vocab_size,
                 cat_vocab_sizes, num_cat_var,
                 num_numerical_var, quantiles,
                 n_heads, d_ff, n_layers, 
                 dropout, d_fc, n_classes, device):
        # super constructor
        super().__init__()
        
        # attributes
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size # vocabulary size
        self.max_len = max_len # text sequence length
        self.cat_vocab_sizes = cat_vocab_sizes # list of vocabulary sizes for categorical variables
        self.num_cat_var = num_cat_var # number of categorical variables
        self.num_numerical_var = num_numerical_var # number of numerical variables
        self.quantiles = quantiles # quantiles for each numerical variable
        self.n_heads = n_heads # number of attention heads
        self.d_ff = d_ff # dimension of the feedforward network model 
        self.n_layers = n_layers # number of encoder layers
        self.dropout = dropout # dropout rate
        self.embedding_dropout = nn.Dropout(dropout) # dropout after text embedding
        self.cat_dropout = nn.Dropout(dropout) # dropout after cat embedding
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.n_classes = n_classes # number of classes
        self.device = device
        
        # text embedding, note that with padding = 0: entries do not contribute to the gradient
        self.text_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.d_model, padding_idx = 0) 
           
        # categorical embeddings
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.cat_vocab_sizes[i], embedding_dim=self.d_model, 
                                            padding_idx = 0) for i in range(self.num_cat_var)])
    
        # embeddings for numericals
        self.num_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=len(self.quantiles[i]), embedding_dim=self.d_model) for i in range(self.num_numerical_var)])
        
        # classification token [CLS], is learnable
        self.text_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        self.tab_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
        
        # positional embedding, not learnable
        self.text_pos_embed = nn.Parameter(data = positional_embeddings(self.max_len + 1, self.d_model), requires_grad = False)
         
        # MM encoder block
        self.MMEncoder_tab_txt = nn.ModuleList([MMEncoderLayer2(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)])
        self.MMEncoder_txt_tab = nn.ModuleList([MMEncoderLayer2(self.d_model, self.n_heads, self.d_ff, self.dropout) for _ in range(self.n_layers)])
        
        # text fully connected network
        self.fc1 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # tabular fully connected network
        self.fc2 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # weight initialization
        self.init_weights()
        
    def init_weights(self):
        # embeddings
        nn.init.kaiming_uniform_(self.text_embedding.weight)
        for i in range(self.num_cat_var):
            nn.init.kaiming_uniform_(self.cat_embeddings[i].weight)
        for i in range(self.num_numerical_var):
            nn.init.kaiming_uniform_(self.num_embeddings[i].weight)
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)
        nn.init.zeros_(self.fc2[0].bias)
        nn.init.kaiming_uniform_(self.fc2[0].weight)
        nn.init.zeros_(self.fc2[3].bias)
        nn.init.kaiming_uniform_(self.fc2[3].weight)
            
    def forward(self, texts, padding_mask, categoricals, numericals):

        # 1. reshape categoricals and numericals for embeddings
        categorical_list = [categoricals[:,i].unsqueeze(dim=1) for i in range(self.num_cat_var)]
        C = torch.zeros((len(numericals), len(self.quantiles[0])), dtype = int).to(self.device)
        for i in range(len(self.quantiles[0])):
            C[:,i] = i
        numerical_token_list = [C for i in range(self.num_numerical_var)]
        weights = [np.abs(np.array([numericals[:,i].cpu().numpy() for j in range(len(self.quantiles[i]))]).T - self.quantiles[i]) for i in range(self.num_numerical_var)] # distance to quantile
        numericals = numericals.to(self.device) 
        weights = [torch.tensor(1/w).to(self.device)  for w in weights] # similarity
        weights = [weights[i]/weights[i].sum(dim=1, keepdim=True) for i in range(self.num_numerical_var)] # normalize
        weights = [torch.nan_to_num(weights[i], 1.) for i in range(self.num_numerical_var)] # replace nan by 1
        weights = [weights[i]/weights[i].sum(dim=1, keepdim=True) for i in range(self.num_numerical_var)] # normalize
        weights = [weights[i].unsqueeze(1) for i in range(self.num_numerical_var)]
        weights = [weights[i].float() for i in range(self.num_numerical_var)]
       
        # 2. embedding layers
        # text embedding
        texts = self.text_embedding(texts) 
        texts = self.embedding_dropout(texts)
        # categorical embedding
        cat_embedding_list = [self.cat_embeddings[i](categorical_list[i]) for i in range(self.num_cat_var)]
        categoricals = torch.cat([cat_embedding_list[i] for i in range(self.num_cat_var)], dim = 1)
        categoricals = self.cat_dropout(categoricals)
        # numerical embedding
        num_embedding_list = [self.num_embeddings[i](numerical_token_list[i]) for i in range(self.num_numerical_var)]
        # numericals: Weights x Quantile Embeddings
        numericals = [torch.bmm(weights[i], num_embedding_list[i]) for i in range(self.num_numerical_var)]
        numericals = torch.cat([numericals[i] for i in range(self.num_numerical_var)], dim=1)
        tabulars = torch.cat([categoricals, numericals], dim = 1) # concatenate categorical and numerical embeddings
        
        # 3. classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        texts = torch.stack([torch.vstack((self.text_cls, texts[i])) for i in range(len(texts))]) * math.sqrt(self.d_model)
        tabulars = torch.stack([torch.vstack((self.tab_cls, tabulars[i])) for i in range(len(tabulars))]) * math.sqrt(self.d_model)
        
        # 4. positional embeddings
        text_pos_embeds = self.text_pos_embed.repeat(len(texts),1,1)
        texts = texts + text_pos_embeds

        ## 5. MM encoder
        texts_dict = {}
        texts_dict[0] = texts
        tabulars_dict = {}
        tabulars_dict[0] = tabulars
        for i, layer in enumerate(self.MMEncoder_tab_txt):
            texts_dict[i+1] = layer(source = tabulars_dict[0], target = [texts_dict[i], padding_mask])
        for i, layer in enumerate(self.MMEncoder_txt_tab):
            tabulars_dict[i+1] = layer(source = [texts_dict[0], padding_mask], target = tabulars_dict[i])
        texts = texts_dict[i+1]
        tabulars = tabulars_dict[i+1]
       
        # 6. Extract CLS tokens
        text_cls = texts[:,0,:]
        tabular_cls = tabulars[:,0,:]
        
        #7. Late fusion (average of logits)
        text_pred = self.fc1(text_cls)
        tabular_pred = self.fc2(tabular_cls)
        pred = (text_pred + tabular_pred)/2 # average
        
        return pred, text_pred, tabular_pred
    
#######################################################################################################################
# 9. LateFuseBERT
#######################################################################################################################

class LateFuseBERT(nn.Module):
    def __init__(self,
                 text_model,
                 cat_vocab_sizes,
                 num_cat_var,
                 num_numerical_var,
                 d_model,
                 n_heads,
                 n_layers, 
                 dropout,
                 d_fc,
                 n_classes):
        # super constructor
        super().__init__()

        # attributes
        self.text_model = text_model
        self.cat_vocab_sizes = cat_vocab_sizes # list of vocabulary sizes for categorical variables
        self.num_cat_var = num_cat_var # number of categorical variables
        self.num_numerical_var = num_numerical_var # number of numerical variables
        self.n_heads = n_heads # number of attention heads
        self.d_model = d_model # embedding dimension
        self.n_layers = n_layers # number of encoder layers
        self.dropout = dropout # dropout rate
        self.cat_dropout = nn.Dropout(dropout) # dropout after cat embedding
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.n_classes = n_classes # number of classes
        
        # categorical embeddings
        self.cat_embeddings = nn.ModuleList([nn.Embedding(num_embeddings=self.cat_vocab_sizes[i], embedding_dim=self.d_model, padding_idx = 0) for i in range(self.num_cat_var)])
    
        # linear mapper for numerical data
        self.num_linears = nn.ModuleList([nn.Linear(1, self.d_model) for i in range(self.num_numerical_var)])
        
        # classification token [CLS], is learnable
        self.tab_cls = nn.Parameter(data=torch.rand(1, self.d_model), requires_grad=True)
          
        # Self Attention Transformer encoder
        self.tab_encoder_layers = nn.TransformerEncoderLayer(self.d_model, self.n_heads, 
                                                              self.d_model, dropout=self.dropout, batch_first=True)
        self.tab_transformer_encoder = nn.TransformerEncoder(self.tab_encoder_layers, self.n_layers)
        
        # last fully connected network
        self.fc1 = nn.Sequential(nn.Linear(2*self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))
        
        # weight initialization
        self.init_weights()

    def init_weights(self):
        # embeddings
        for i in range(self.num_cat_var):
            nn.init.kaiming_uniform_(self.cat_embeddings[i].weight)
        # numerical linear
        for i in range(self.num_numerical_var):
            nn.init.zeros_(self.num_linears[i].bias)
            nn.init.kaiming_uniform_(self.num_linears[i].weight)
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)


    def forward(self, texts, attention_mask, categoricals, numericals):

        # 1. reshape categoricals for embeddings and numericals before linear transformation 
        categorical_list = [categoricals[:,i].unsqueeze(dim=1) for i in range(self.num_cat_var)]
        numerical_list = [numericals[:,i].unsqueeze(dim=1).unsqueeze(dim=1) for i in range(self.num_numerical_var)]
        
        # 2. embedding layers
        cat_embedding_list = [self.cat_embeddings[i](categorical_list[i]) for i in range(self.num_cat_var)]
        categoricals = torch.cat([cat_embedding_list[i] for i in range(self.num_cat_var)], dim = 1)
        categoricals = self.cat_dropout(categoricals)
        numerical_embedding_list = [self.num_linears[i](numerical_list[i].float()) for i in range(self.num_numerical_var)]
        numericals = torch.cat([numerical_embedding_list[i] for i in range(self.num_numerical_var)], dim = 1)
        tabulars = torch.cat([categoricals, numericals], dim = 1) # concatenate categorical and numerical embeddings
        
        # 3. add classification token [CLS], * sqrt(d) prevent these input embeddings from becoming excessively small
        tabulars = torch.stack([torch.vstack((self.tab_cls, tabulars[i])) for i in range(len(tabulars))]) * math.sqrt(self.d_model)
        
        # 4. Self attention Transformer encoder (tabular stream)
        tabulars = self.tab_transformer_encoder(tabulars)
   
        # 5. text model prediction
        texts = self.text_model(texts, attention_mask = attention_mask).last_hidden_state

        # 6. Concatenate CLS tokens
        text_cls = texts[:,0,:]
        tabular_cls = tabulars[:,0,:]
        mm_cls = torch.cat([text_cls, tabular_cls], dim = 1)

        # 7. Fully connected network for classification purpose
        pred = self.fc1(mm_cls)
        
        return pred, text_cls, tabular_cls

#######################################################################################################################
# 10. AllTextBERT
#######################################################################################################################

class AllTextBERT(nn.Module):
    def __init__(self,
                 text_model,
                 d_model,
                 dropout,
                 d_fc,
                 n_classes):
        # super constructor
        super().__init__()

        # attributes
        self.text_model = text_model
        self.d_model = d_model
        self.d_fc = d_fc # dimension of hidden layer in final fully connected layer
        self.dropout = dropout # dropout rate
        self.n_classes = n_classes # number of classes
        
        # last fully connected network
        self.fc1 = nn.Sequential(nn.Linear(self.d_model, self.d_fc),
                                nn.ReLU(),
                                nn.Dropout(self.dropout),
                                nn.Linear(self.d_fc, self.n_classes))

        
        # weight initialization
        self.init_weights()

    def init_weights(self):
        # final FC network
        nn.init.zeros_(self.fc1[0].bias)
        nn.init.kaiming_uniform_(self.fc1[0].weight)
        nn.init.zeros_(self.fc1[3].bias)
        nn.init.kaiming_uniform_(self.fc1[3].weight)

    def forward(self, texts, attention_mask, categoricals, numericals):

        # 1. text model prediction
        texts = self.text_model(texts, attention_mask = attention_mask).last_hidden_state

        # 2. Extract CLS tokens
        text_cls = texts[:,0,:]
        
        # 3. Logits
        text_pred = self.fc1(text_cls)

        return text_pred, text_cls    
            
##############################################################################################################

def init_model(model_type, d_model, max_len, vocab_size, cat_vocab_sizes, 
               num_cat_var, num_numerical_var, quantiles, n_heads,
               d_ff, n_layers, dropout, d_fc, n_classes, seed, device, text_model=""):
                   
    if model_type in ["TTT-SRP", "TTT-PCA", "TTT-Kaiming"]:
        model_type = "TTT"

    if model_type == "MulT":
        torch.manual_seed(seed)
        model =  MMTranformer(d_model, 
                              max_len, 
                              vocab_size,
                              cat_vocab_sizes, 
                              num_cat_var,
                              num_numerical_var,
                              n_heads,
                              d_ff, 
                              n_layers, 
                              dropout,
                              d_fc,
                              n_classes)

    if model_type == "EarlyConcat":
        torch.manual_seed(seed)
        model =  EarlyConcatTranformer(d_model, 
                              max_len, 
                              vocab_size,
                              cat_vocab_sizes, 
                              num_cat_var,
                              num_numerical_var,
                              n_heads,
                              d_ff, 
                              n_layers, 
                              dropout,
                              d_fc,
                              n_classes) 

    if model_type == "LateFuse":
        torch.manual_seed(seed)
        model =  TabularTranformer(d_model, 
                              max_len, 
                              vocab_size,
                              num_cat_var,
                              num_numerical_var,
                              n_heads,
                              d_ff, 
                              n_layers, 
                              dropout,
                              d_fc,
                              n_classes)


    if model_type == "TFN":
        torch.manual_seed(seed)
        model =  TFN(d_model, 
                              vocab_size,
                              num_cat_var,
                              num_numerical_var,
                              dropout,
                              d_fc,
                              n_classes,
                              device)


    if model_type == "TTT":
        torch.manual_seed(seed)
        model =  TTT(d_model, 
                              max_len, 
                              vocab_size,
                              cat_vocab_sizes, 
                              num_cat_var,
                              num_numerical_var,
                              quantiles,
                              n_heads,
                              d_ff, 
                              n_layers, 
                              dropout,
                              d_fc,
                              n_classes,
                              device)

    if model_type == "TTT_ablation1":
        torch.manual_seed(seed)
        model =  TTT_ablation1(d_model, 
                                max_len, 
                                vocab_size,
                                cat_vocab_sizes, 
                                num_cat_var,
                                num_numerical_var,
                                n_heads, 
                                d_ff, 
                                n_layers, 
                                dropout, 
                                d_fc, 
                                n_classes, 
                                device)
                                
    if model_type == "TTT_ablation2":
        torch.manual_seed(seed)
        model =  TTT_ablation2(d_model, 
                              max_len, 
                              vocab_size,
                              cat_vocab_sizes, 
                              num_cat_var,
                              num_numerical_var,
                              quantiles,
                              n_heads,
                              d_ff, 
                              n_layers, 
                              dropout,
                              d_fc,
                              n_classes,
                              device)
                                
    if model_type == "TTT_ablation3":
        torch.manual_seed(seed)
        model =  TTT_ablation3(d_model, 
                              max_len, 
                              vocab_size,
                              cat_vocab_sizes, 
                              num_cat_var,
                              num_numerical_var,
                              quantiles,
                              n_heads,
                              d_ff, 
                              n_layers, 
                              dropout,
                              d_fc,
                              n_classes,
                              device)
                              
    if model_type == "LateFuseBERT":
        torch.manual_seed(seed)
        model = LateFuseBERT(text_model,
                             cat_vocab_sizes,
                             num_cat_var,
                             num_numerical_var,
                             d_model,
                             n_heads,
                             n_layers, 
                             dropout,
                             d_fc,
                             n_classes)
                                 
    if model_type == "AllTextBERT":
        torch.manual_seed(seed)
        model = AllTextBERT(text_model,
                            d_model,
                            dropout,
                            d_fc,
                            n_classes)
                   
    
    return model
