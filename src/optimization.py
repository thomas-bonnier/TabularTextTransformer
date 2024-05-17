from torch.optim.lr_scheduler import ExponentialLR
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset, DataLoader
from model import * # models
from settings import * # settings
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
from sklearn.decomposition import PCA
from sklearn.random_projection import SparseRandomProjection


# class for model training
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience):
        """
        Arg: patience (int): How long to wait after last time validation loss improved.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_performance, model):

        if self.best_score is None:
            self.best_score = val_performance
            self.save_checkpoint(model)
        
        elif val_performance < (self.best_score)*1.0001 :
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_performance
            self.save_checkpoint(model)
            self.counter = 0

    
    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), 'checkpoint.pt')

# Model's performance evaluation
def evaluation(model, loader, criterion, model_type, seed, device):
    """
    Args: pytorch model, pytorch loader, seed
    Returns evaluation metrics
    """
    if model_type in ["TTT-SRP", "TTT-PCA", "TTT-Kaiming"]:
        model_type = "TTT"
    total = 0
    loss= 0 
    torch.manual_seed(seed) 
    for text, categorical, numerical, y in loader:
        model.eval()
        if model_type in ["MulT", "LateFuse", "TTT_ablation2"]:
            text_with_ones = torch.column_stack((torch.ones((y.shape[0],1), dtype=int), text)) # add one column for CLS token
        if model_type in ["EarlyConcat"]:
            text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 1), dtype=int), text)) # add one column for CLS token
        if model_type in ["TTT", "TTT_ablation1", "TTT_ablation3"]:  
            text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 2), dtype=int), text)) 
        if model_type in ["TFN"]: 
            text_with_ones = torch.ones((1,1))
        padding_mask = text_with_ones==0
        text = text.to(device)
        padding_mask = padding_mask.to(device)
        categorical = categorical.to(device)
        numerical = numerical.to(device)
        y = y.to(device)             
        with torch.no_grad():
            if model_type in ["TTT", "TTT_ablation1", "TTT_ablation2", "TTT_ablation3"]:
                y_hat = model(text, padding_mask, categorical, numerical)[0]
            else:
                y_hat = model(text, padding_mask, categorical, numerical)
            
        total += y.shape[0]
        loss += criterion(y_hat,y).item()*y.shape[0]
    return loss/total

# function for training the model
def training(model, loader_train,  n_epochs, loader_validation, criterion, optimizer, patience, factor, model_type, seed, verbose, device, early_stopping):
    
    if model_type in ["TTT-SRP", "TTT-PCA", "TTT-Kaiming"]:
        model_type = "TTT"
        
    if early_stopping:
        # early stopping
        early_stopping = EarlyStopping(patience = patience)
    # scheduler
    scheduler = ExponentialLR(optimizer, gamma=factor)
    
    # training and validation loop
    for epoch in range(1, n_epochs+1):
        start=time.time()
        train_loss = 0 # training loss by sample
        total = 0 # number of samples
        torch.manual_seed(seed)
        for text, categorical, numerical, y in loader_train:
            model.train()
            # 1. clear gradients
            optimizer.zero_grad()
            
            # 2. key padding mask
            if model_type in ["MulT", "LateFuse", "TTT_ablation2"]:
                text_with_ones = torch.column_stack((torch.ones((y.shape[0],1), dtype=int), text)) # add one column for CLS token
            if model_type in ["EarlyConcat"]:
                text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 1), dtype=int), text)) # add one column for CLS token
            if model_type in ["TTT", "TTT_ablation1", "TTT_ablation3"]:  
                text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 2), dtype=int), text)) 
            if model_type in ["TFN"]: 
                text_with_ones = torch.ones((1,1))
            padding_mask = text_with_ones==0
            text = text.to(device)
            padding_mask = padding_mask.to(device)
            categorical = categorical.to(device)
            numerical = numerical.to(device)
            y = y.to(device)
            
            # 3. forward pass and compute loss
            if model_type in ["TTT", "TTT_ablation1", "TTT_ablation2"]:
                y_hat, text_pred, tabular_pred = model(text, padding_mask, categorical, numerical.float()) 
                loss = criterion(text_pred,y)+criterion(tabular_pred,y) # dual loss 
            elif model_type in ["TTT_ablation3"]:
                y_hat, text_pred, tabular_pred = model(text, padding_mask, categorical, numerical.float())
                loss = criterion(y_hat,y)
            else:
                y_hat = model(text, padding_mask, categorical, numerical)
                loss = criterion(y_hat,y)
                
            # 4. backward pass
            loss.backward()
            # 5. optimization
            optimizer.step()
            # 6. record loss
            train_loss += loss.item()*y.shape[0]
            total += y.shape[0]

        end=time.time()   
        train_loss = train_loss/total
        if verbose:
            print("---------training time (s):", round(end-start,0), "---------")
            print("epoch:", epoch, "training loss:", round(train_loss,5))

        # model's performance evaluation
        val_loss = evaluation(model, loader_validation, criterion, model_type, seed, device)
        if model_type in ["TTT", "TTT_ablation1", "TTT_ablation2", "TTT_ablation3"]:
            validation_performance = performance(model, loader_validation, model_type, seed, device)[0]
        else:
            validation_performance = performance(model, loader_validation, model_type, seed, device)
        # scheduler.step(validation_performance)
        scheduler.step()

        if verbose:
            print("epoch:", epoch, "validation loss:", round(val_loss,5), "validation performance:", round(validation_performance,5))
        
        if early_stopping:
            # early stopping update and condition
            early_stopping(validation_performance, model)
            if early_stopping.early_stop:
                break
            
    # load the last checkpoint with the best model        
    if early_stopping:
        model.load_state_dict(torch.load('checkpoint.pt'))
        
    return model
        
# performance computation
def performance(model, loader_target, model_type, seed, device):
    """Performance computation (accuracy)"""
    
    if model_type in ["TTT-SRP", "TTT-PCA", "TTT-Kaiming"]:
        model_type = "TTT"
        
    preds_list = []
    text_preds_list = []
    tabular_preds_list = []
    labels_list = []
    torch.manual_seed(seed)
    for text, categorical, numerical, y in loader_target:
        # evaluation mode
        model.eval()
        # key padding mask
        if model_type in ["MulT", "LateFuse", "TTT_ablation2"]:
            text_with_ones = torch.column_stack((torch.ones((y.shape[0],1), dtype=int), text)) # add one column for CLS token
        if model_type in ["EarlyConcat"]:
            text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 1), dtype=int), text)) # add one column for CLS token
        if model_type in ["TTT", "TTT_ablation1", "TTT_ablation3"]:  
            text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 2), dtype=int), text)) 
        if model_type in ["TFN"]: 
            text_with_ones = torch.ones((1,1))
        padding_mask = text_with_ones==0
        text = text.to(device)
        padding_mask = padding_mask.to(device)
        categorical = categorical.to(device)
        numerical = numerical.to(device)
        y = y.to(device)
        # prediction
        with torch.no_grad():
            if model_type in ["TTT", "TTT_ablation1", "TTT_ablation2", "TTT_ablation3"]:
                pred, text_pred, tabular_pred = model(text, padding_mask, categorical, numerical.float())
            else:
                pred = model(text, padding_mask, categorical, numerical.float())
        # compute softmax probabilities
        p_hat = F.softmax(pred, dim=1)  
        preds_list.append(p_hat)
        if model_type in ["TTT", "TTT_ablation1", "TTT_ablation2", "TTT_ablation3"]: # unimodal preds
            p_hat_text = F.softmax(text_pred, dim=1)
            p_hat_tab = F.softmax(tabular_pred, dim=1)
            text_preds_list.append(p_hat_text)
            tabular_preds_list.append(p_hat_tab)
            
        labels_list.append(y)

    labels_list = torch.cat(labels_list)
    preds_list = torch.cat(preds_list)
    
    if model_type in ["TTT", "TTT_ablation1", "TTT_ablation2", "TTT_ablation3"]:
        text_preds_list = torch.cat(text_preds_list)
        tabular_preds_list = torch.cat(tabular_preds_list)
        text_preds = torch.argmax(text_preds_list, dim=1)
        tabular_preds = torch.argmax(tabular_preds_list, dim=1)
        
    performance = sum(torch.argmax(preds_list, dim=1)==labels_list).item()/labels_list.shape[0] # accuracy
    
    if model_type in ["TTT", "TTT_ablation1", "TTT_ablation2", "TTT_ablation3"]:
        return performance, labels_list.cpu().numpy(), torch.argmax(preds_list, dim=1).cpu().numpy(), text_preds.cpu().numpy(), tabular_preds.cpu().numpy()

    else:
        return performance
    
# model optimization and selection
def hp_optimization(model_type, dataset_train, dataset_validation,
max_len, vocab_size, cat_vocab_sizes, num_cat_var, num_numerical_var, quantiles,
criterion, seed, device, dataset):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # settings
    EPOCHS = 5

    FILENAME, categorical_var, numerical_var, text_var, MAX_LEN_QUANTILE, N_CLASSES, WEIGHT_DECAY, FACTOR, N_EPOCHS, split_val, CRITERION, N_SEED, DROPOUT= load_settings(dataset)
    
    MAX_LEN=max_len
    VOCAB_SIZE=vocab_size
    CAT_VOCAB_SIZES=cat_vocab_sizes
    NUM_CAT_VAR=num_cat_var
    NUM_NUMERICAL_VAR=num_numerical_var
    QUANTILES=quantiles
    
    # define model
    def define_model(trial):
        
        # optimization for the model: embedding dim, number of attention heads and number of layers
        D_MODEL = trial.suggest_int("D_MODEL", 32, 64, step = 16)
        N_LAYERS = trial.suggest_int("N_LAYERS", 2, 3)
        N_HEADS = trial.suggest_int("N_HEADS", 4, 8, step = 4)
        
        # same dimension for Feed Forward and Fully Connected
        D_FF = D_MODEL        
        D_FC = D_MODEL

        if model_type == "MulT":
            torch.manual_seed(seed)
            model =  MMTranformer(d_model = D_MODEL, 
                                  max_len = MAX_LEN, 
                                  vocab_size = VOCAB_SIZE,
                                  cat_vocab_sizes = CAT_VOCAB_SIZES, 
                                  num_cat_var = NUM_CAT_VAR,
                                  num_numerical_var = NUM_NUMERICAL_VAR,
                                  n_heads = N_HEADS,
                                  d_ff = D_FF, 
                                  n_layers = N_LAYERS, 
                                  dropout = DROPOUT,
                                  d_fc = D_FC,
                                  n_classes = N_CLASSES).to(device)

        if model_type == "EarlyConcat":
            torch.manual_seed(seed)
            model =  EarlyConcatTranformer(d_model = D_MODEL, 
                                  max_len = MAX_LEN, 
                                  vocab_size = VOCAB_SIZE,
                                  cat_vocab_sizes = CAT_VOCAB_SIZES, 
                                  num_cat_var = NUM_CAT_VAR,
                                  num_numerical_var = NUM_NUMERICAL_VAR,
                                  n_heads = N_HEADS,
                                  d_ff = D_FF, 
                                  n_layers = N_LAYERS, 
                                  dropout = DROPOUT,
                                  d_fc = D_FC,
                                  n_classes = N_CLASSES).to(device)

        if model_type == "LateFuse":
            torch.manual_seed(seed)
            model =  TabularTranformer(d_model = D_MODEL, 
                                  max_len = MAX_LEN, 
                                  vocab_size = VOCAB_SIZE,
                                  num_cat_var = NUM_CAT_VAR,
                                  num_numerical_var = NUM_NUMERICAL_VAR,
                                  n_heads = N_HEADS,
                                  d_ff = D_FF, 
                                  n_layers = N_LAYERS, 
                                  dropout = DROPOUT,
                                  d_fc = D_FC,
                                  n_classes = N_CLASSES).to(device)


        if model_type == "TFN":
            torch.manual_seed(seed)
            model =  TFN(d_model = D_MODEL, 
                                  vocab_size = VOCAB_SIZE,
                                  num_cat_var = NUM_CAT_VAR,
                                  num_numerical_var = NUM_NUMERICAL_VAR,
                                  dropout = DROPOUT,
                                  d_fc = D_FC,
                                  n_classes = N_CLASSES,
                                  device=device).to(device)


        if model_type == "TTT":
            torch.manual_seed(seed)
            model =  TTT(d_model = D_MODEL, 
                                  max_len = MAX_LEN, 
                                  vocab_size = VOCAB_SIZE,
                                  cat_vocab_sizes = CAT_VOCAB_SIZES, 
                                  num_cat_var = NUM_CAT_VAR,
                                  num_numerical_var = NUM_NUMERICAL_VAR,
                                  quantiles = QUANTILES,
                                  n_heads = N_HEADS,
                                  d_ff = D_FF, 
                                  n_layers = N_LAYERS, 
                                  dropout = DROPOUT,
                                  d_fc = D_FC,
                                  n_classes = N_CLASSES,
                                  device=device).to(device)
                                  
        if model_type == "TTT_ablation1":
            torch.manual_seed(seed)
            model =  TTT_ablation1(d_model = D_MODEL, 
                                  max_len = MAX_LEN, 
                                  vocab_size = VOCAB_SIZE,
                                  cat_vocab_sizes = CAT_VOCAB_SIZES, 
                                  num_cat_var = NUM_CAT_VAR,
                                  num_numerical_var = NUM_NUMERICAL_VAR,
                                  n_heads = N_HEADS,
                                  d_ff = D_FF, 
                                  n_layers = N_LAYERS, 
                                  dropout = DROPOUT,
                                  d_fc = D_FC,
                                  n_classes = N_CLASSES,
                                  device=device).to(device)

        if model_type == "TTT_ablation2":
            torch.manual_seed(seed)
            model =  TTT_ablation2(d_model = D_MODEL, 
                                  max_len = MAX_LEN, 
                                  vocab_size = VOCAB_SIZE,
                                  cat_vocab_sizes = CAT_VOCAB_SIZES, 
                                  num_cat_var = NUM_CAT_VAR,
                                  num_numerical_var = NUM_NUMERICAL_VAR,
                                  quantiles = QUANTILES,
                                  n_heads = N_HEADS,
                                  d_ff = D_FF, 
                                  n_layers = N_LAYERS, 
                                  dropout = DROPOUT,
                                  d_fc = D_FC,
                                  n_classes = N_CLASSES,
                                  device=device).to(device) 
                                  
        if model_type == "TTT_ablation3":
            torch.manual_seed(seed)
            model =  TTT_ablation3(d_model = D_MODEL, 
                                  max_len = MAX_LEN, 
                                  vocab_size = VOCAB_SIZE,
                                  cat_vocab_sizes = CAT_VOCAB_SIZES, 
                                  num_cat_var = NUM_CAT_VAR,
                                  num_numerical_var = NUM_NUMERICAL_VAR,
                                  quantiles = QUANTILES,
                                  n_heads = N_HEADS,
                                  d_ff = D_FF, 
                                  n_layers = N_LAYERS, 
                                  dropout = DROPOUT,
                                  d_fc = D_FC,
                                  n_classes = N_CLASSES,
                                  device=device).to(device)

        return model
    
    def objective(trial):
        # generate the model
        model = define_model(trial)
        
        # optimizer
        LR = trial.suggest_loguniform('LR', 1e-5, 1e-3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # batch size
        BATCH_SIZE = trial.suggest_int("BATCH_SIZE", 32, 128, step = 32)
        
        # loaders
        loader_train = DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True)
        loader_validation = DataLoader(dataset_validation, batch_size = BATCH_SIZE, shuffle = True)
        
        # we keep only part of training set to make it faster
        if len(dataset_train)>20000:
            N_TRAIN_EXAMPLES = int(len(dataset_train)*0.6)
        else:
            N_TRAIN_EXAMPLES = int(len(dataset_train))
        N_TRAIN_EXAMPLES = N_TRAIN_EXAMPLES // BATCH_SIZE
        N_TRAIN_EXAMPLES = N_TRAIN_EXAMPLES * BATCH_SIZE

        # Training of the model.
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (text, categorical, numerical, y) in enumerate(loader_train):
                # Limiting training data for faster epochs.
                if batch_idx * BATCH_SIZE >= N_TRAIN_EXAMPLES:
                    break
                # clear gradients
                optimizer.zero_grad()
                # key padding mask
                if model_type in ["MulT", "LateFuse", "TTT_ablation2"]:
                    text_with_ones = torch.column_stack((torch.ones((y.shape[0],1), dtype=int), text)) # add one column for CLS token
                if model_type in ["EarlyConcat"]:
                    text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 1), dtype=int), text)) # add one column for CLS token
                if model_type in ["TTT", "TTT_ablation1", "TTT_ablation3"]:  
                    text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 2), dtype=int), text)) 
                if model_type in ["TFN"]: 
                    text_with_ones = torch.ones((1,1))
                padding_mask = text_with_ones==0
                text = text.to(device)
                padding_mask = padding_mask.to(device)
                categorical = categorical.to(device)
                numerical = numerical.to(device)
                y = y.to(device)
    
                # forward pass and compute loss
                if model_type in ["TTT", "TTT_ablation1", "TTT_ablation2"]:
                    y_hat, text_pred, tabular_pred = model(text, padding_mask, categorical, numerical.float())
                    loss = criterion(text_pred,y)+criterion(tabular_pred,y)
                elif model_type in ["TTT_ablation3"]:
                    y_hat, text_pred, tabular_pred = model(text, padding_mask, categorical, numerical.float())
                    loss = criterion(y_hat,y)
                else:
                    y_hat = model(text, padding_mask, categorical, numerical)
                    loss = criterion(y_hat,y)
                # backward pass
                loss.backward()
                # optimization
                optimizer.step()
    
            # Validation of the model: model's performance evaluation
            if model_type in ["TTT", "TTT_ablation1", "TTT_ablation2", "TTT_ablation3"]:
                accuracy = performance(model, loader_validation, model_type, seed, device)[0]
            else:
                accuracy = performance(model, loader_validation, model_type, seed, device)
   
            trial.report(accuracy, epoch)
    
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
        return accuracy
        
    # optimize
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed)  # TPE sampler behaves in a deterministic way
    study = optuna.create_study(direction="maximize", sampler = sampler)
    if len(dataset_train)>20000:
        timeout = 900
    else:
        timeout = 600

    study.optimize(objective, timeout = timeout) # 10 or 15 minutes
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    # select best trial
    trial = study.best_trial

    return trial.params
 
# model optimization and selection
def hp_optimization_large(model_type, dataset_train, dataset_validation, text_model, init_weights,
max_len, vocab_size, cat_vocab_sizes, num_cat_var, num_numerical_var, quantiles,
criterion, seed, device, dataset):
    
    if model_type in ["TTT-SRP", "TTT-PCA", "TTT-Kaiming"]:
        model_type = "TTT"
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # settings
    EPOCHS = 5

    FILENAME, categorical_var, numerical_var, text_var, MAX_LEN_QUANTILE, N_CLASSES, WEIGHT_DECAY, FACTOR, N_EPOCHS, split_val, CRITERION, N_SEED, DROPOUT= load_settings(dataset)
    
    MAX_LEN=max_len
    VOCAB_SIZE=vocab_size
    CAT_VOCAB_SIZES=cat_vocab_sizes
    NUM_CAT_VAR=num_cat_var
    NUM_NUMERICAL_VAR=num_numerical_var
    QUANTILES=quantiles
    
    # define model
    def define_model(trial):
        
        # optimization for the model: embedding dim, number of attention heads and number of layers
        D_MODEL = trial.suggest_int("D_MODEL", 120, 240, step = 24)
        N_LAYERS = trial.suggest_int("N_LAYERS", 4, 5)
        N_HEADS = trial.suggest_int("N_HEADS", 4, 12, step = 4)
        
        # same dimension for Feed Forward and Fully Connected
        D_FF = D_MODEL        
        D_FC = D_MODEL



        if model_type == "TTT":
            torch.manual_seed(seed)
            model =  TTT(d_model = D_MODEL, 
                                  max_len = MAX_LEN, 
                                  vocab_size = VOCAB_SIZE,
                                  cat_vocab_sizes = CAT_VOCAB_SIZES, 
                                  num_cat_var = NUM_CAT_VAR,
                                  num_numerical_var = NUM_NUMERICAL_VAR,
                                  quantiles = QUANTILES,
                                  n_heads = N_HEADS,
                                  d_ff = D_FF, 
                                  n_layers = N_LAYERS, 
                                  dropout = DROPOUT,
                                  d_fc = D_FC,
                                  n_classes = N_CLASSES,
                                  device=device).to(device)
            
            if init_weights != "kaiming":

                # Load embeddings weights from pretrained model
                pretrained_dict = text_model.state_dict()
                model_dict = model.state_dict()
                updated_model_dict = {k: v for k, v in model_dict.items()}
                distil_bert_embeddings = text_model.state_dict()['embeddings.word_embeddings.weight']

                # dimension reduction
                if init_weights == "pca":
                    reduction_technique = PCA(n_components=D_MODEL, random_state = seed)
                if init_weights == "random_proj":
                    reduction_technique = SparseRandomProjection(n_components=D_MODEL, random_state = seed)
                reduced_embeddings = torch.tensor(reduction_technique.fit_transform(distil_bert_embeddings.cpu().numpy())).to(device)

                # update embeddings
                updated_model_dict.update([('text_embedding.weight', reduced_embeddings)])
                model_dict.update(updated_model_dict)
                model.load_state_dict(model_dict)
                    
                                    
        return model
    
    def objective(trial):
        # generate the model
        model = define_model(trial)
        
        # optimizer
        LR = trial.suggest_loguniform('LR', 1e-5, 1e-3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        
        # batch size
        BATCH_SIZE = trial.suggest_int("BATCH_SIZE", 32, 128, step = 32)
        
        # loaders
        loader_train = DataLoader(dataset_train, batch_size = BATCH_SIZE, shuffle = True)
        loader_validation = DataLoader(dataset_validation, batch_size = BATCH_SIZE, shuffle = True)
        
        # we keep only part of training set to make it faster
        if len(dataset_train)>20000:
            N_TRAIN_EXAMPLES = int(len(dataset_train)*0.6)
        else:
            N_TRAIN_EXAMPLES = int(len(dataset_train))
        N_TRAIN_EXAMPLES = N_TRAIN_EXAMPLES // BATCH_SIZE
        N_TRAIN_EXAMPLES = N_TRAIN_EXAMPLES * BATCH_SIZE

        # Training of the model.
        for epoch in range(EPOCHS):
            model.train()
            for batch_idx, (text, categorical, numerical, y) in enumerate(loader_train):
                # Limiting training data for faster epochs.
                if batch_idx * BATCH_SIZE >= N_TRAIN_EXAMPLES:
                    break
                # clear gradients
                optimizer.zero_grad()
                # key padding mask
                if model_type in ["TTT"]:  
                    text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 2), dtype=int), text)) 
                padding_mask = text_with_ones==0
                text = text.to(device)
                padding_mask = padding_mask.to(device)
                categorical = categorical.to(device)
                numerical = numerical.to(device)
                y = y.to(device)
    
                # forward pass and compute loss
                if model_type in ["TTT"]:
                    y_hat, text_pred, tabular_pred = model(text, padding_mask, categorical, numerical.float())
                    loss = criterion(text_pred,y)+criterion(tabular_pred,y)
            
                # backward pass
                loss.backward()
                # optimization
                optimizer.step()
    
            # Validation of the model: model's performance evaluation
            if model_type in ["TTT"]:
                accuracy = performance(model, loader_validation, model_type, seed, device)[0]

            trial.report(accuracy, epoch)
    
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    
        return accuracy
        
    # optimize
    # optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=seed)  # TPE sampler behaves in a deterministic way
    study = optuna.create_study(direction="maximize", sampler = sampler)
    if len(dataset_train)>20000:
        timeout = 900
    else:
        timeout = 600

    study.optimize(objective, timeout = timeout) # 10 or 15 minutes
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    # select best trial
    trial = study.best_trial

    return trial.params
  


# Model's performance evaluation
def evaluation_pretrained(model, loader, criterion, seed, device):
    """
    Args: pytorch model, pytorch loader, seed
    Returns evaluation metrics
    """
    total = 0
    loss= 0 
    torch.manual_seed(seed) 
    for batch  in loader:
        model.eval()
        text = batch[0]
        categorical = batch[1]
        numerical = batch[2]
        y = batch[3]
        mask = batch[4]
        # to device
        text = text.to(device)
        mask = mask.to(device)
        categorical = categorical.to(device)
        numerical = numerical.to(device)
        y = y.to(device) 
        # predict
        with torch.no_grad():
            y_hat = model(text, mask, categorical, numerical)[0]
        # compute loss
        total += y.shape[0]
        loss += criterion(y_hat,y).item()*y.shape[0]
    return loss/total

# function for training the model
def training_pretrained(model, model_type, loader_train,  n_epochs, loader_validation, criterion, optimizer, factor, seed, verbose, device):
    
    # tracking variable
    best_val_perf = 0

    # scheduler
    scheduler = ExponentialLR(optimizer, gamma=factor)
    
    # training and validation loop
    for epoch in range(1, n_epochs+1):
        start=time.time()
        train_loss = 0 # training loss by sample
        total = 0 # number of samples
        torch.manual_seed(seed)
        for batch  in loader_train:
            model.train()
            text = batch[0]
            categorical = batch[1]
            numerical = batch[2]
            y = batch[3]
            mask = batch[4]
            
            # 1. clear gradients
            optimizer.zero_grad()
            
            # 2. to device
            text = text.to(device)
            mask = mask.to(device)
            categorical = categorical.to(device)
            numerical = numerical.to(device)
            y = y.to(device)
            
            # 3. forward pass and compute loss
            y_hat = model(text, mask, categorical, numerical)[0]
            loss = criterion(y_hat,y)
                
            # 4. backward pass
            loss.backward()
            
            # 5. optimization
            optimizer.step()
            
            # 6. record loss
            train_loss += loss.item()*y.shape[0]
            total += y.shape[0]

        end=time.time()   
        train_loss = train_loss/total
        if verbose:
            print("---------training time (s):", round(end-start,0), "---------")
            print("epoch:", epoch, "training loss:", round(train_loss,5))

        # model's performance evaluation (accuracy)
        val_loss = evaluation_pretrained(model, loader_validation, criterion, seed, device)
        validation_performance = performance_pretrained(model, loader_validation, model_type, seed, device)
        
        # scheduler step
        scheduler.step()

        if verbose:
            print("epoch:", epoch, "validation loss:", round(val_loss,5), "validation performance:", round(validation_performance,5))
            
        # save best model so far
        if validation_performance > best_val_perf*1.001: # increase in accuracy should be greater than 0.1%
            torch.save(model, 'checkpoint.pt')
            best_val_perf = validation_performance
        else: # ends training
            break
            
    # load the last checkpoint with the best model        
    model = torch.load("checkpoint.pt")
        
    return model, epoch
        
# performance computation
def performance_pretrained(model, loader_target, model_type, seed, device):
    """Performance computation (accuracy)"""
    preds_list = []
    text_preds_list = []
    tabular_preds_list = []
    labels_list = []
    torch.manual_seed(seed)
    for batch in loader_target:
        # evaluation mode
        model.eval()
        # inputs and labels
        text = batch[0]
        categorical = batch[1]
        numerical = batch[2]
        y = batch[3]
        mask = batch[4]
        # to device
        text = text.to(device)
        mask = mask.to(device)
        categorical = categorical.to(device)
        numerical = numerical.to(device)
        y = y.to(device)
        # prediction
        with torch.no_grad():
            pred = model(text, mask, categorical, numerical.float())[0]
        # compute softmax probabilities
        p_hat = F.softmax(pred, dim=1)  
        preds_list.append(p_hat)
        labels_list.append(y)

    labels_list = torch.cat(labels_list)
    preds_list = torch.cat(preds_list)

        
    performance = sum(torch.argmax(preds_list, dim=1)==labels_list).item()/labels_list.shape[0] # accuracy
    
    return performance    
    
    



        
  

  
