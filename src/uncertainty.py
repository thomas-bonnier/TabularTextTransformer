from dataset import preprocess_dataset
import re
import numbers
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LogisticRegression
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr
from scipy.special import binom
from scipy.stats import entropy

def value_function_jsd(loader, model, model_type, seed, device):
    """
    Returns the softmax and predictions of each stream for the given loader and model.
    This function is used to assess the feature contribution in the approximation of Shapley values with Jensen Shannon Distance.
    """
    text_pred_list = []
    tabular_pred_list = []
    pred_list = []
    torch.manual_seed(seed)
    for batch in loader:
      # evaluation mode
      model.eval()

      # extract inputs and labels
      text = batch[0]
      categorical = batch[1]
      numerical = batch[2]
      y = batch[3]

      # key padding mask
      if model_type in ["TTT"]:
            text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 2), dtype=int), text))
            mask = text_with_ones==0

      # put to device
      text = text.to(device)
      mask = mask.to(device)
      categorical = categorical.to(device)
      numerical = numerical.to(device)
      y = y.to(device)

      # prediction and MM representation
      with torch.no_grad():
          pred, text_pred, tabular_pred  = model(text, mask, categorical, numerical.float())

      # add predictions to lists
      pred_list.append(pred)
      text_pred_list.append(text_pred)
      tabular_pred_list.append(tabular_pred)

    # concatenate and compute softmax
    pred_list = torch.cat(pred_list)
    preds = torch.argmax(pred_list, dim=1)
    text_pred_list = torch.cat(text_pred_list)
    text_preds = torch.argmax(text_pred_list, dim=1)
    text_softs = F.softmax(text_pred_list, dim=1)
    tabular_pred_list = torch.cat(tabular_pred_list)
    tabular_preds = torch.argmax(tabular_pred_list, dim=1)
    tabular_softs = F.softmax(tabular_pred_list, dim=1)

    # compute JSD between text and tabular softmax distributions
    jsd = jensenshannon(text_softs.cpu().numpy(),tabular_softs.cpu().numpy(), axis=1, base=2)
    if str(jsd[0])=='nan':
        jsd = np.array([0.])

    return text_preds.cpu().numpy(), text_softs.cpu().numpy(), tabular_preds.cpu().numpy(), tabular_softs.cpu().numpy(), jsd, preds.cpu().numpy()

def uncertainty_classifier(X, y, seed):
    """
    Value function based on uncertainty classifier
    """

    uncertainty_classifier = RandomForestClassifier(n_estimators=100, random_state=seed)
    uncertainty_classifier.fit(X, y)

    return uncertainty_classifier

def plot_shapley(shapley_result, type="jsd", adjusted = False):
    """plot shapley values"""

    if adjusted:
        var = "Shapley value" + " (" + type + ")-adjusted"
    else:
        var = "Shapley value" + " (" + type + ")"

    # plot representation for top 10 contribution
    shapley_result["Abs Shapley value"] = np.abs(shapley_result[var])
    shapley_top_10 = shapley_result.sort_values(["Abs Shapley value"], ascending = False)[:10]
    for i in shapley_top_10.index:
        if shapley_top_10.loc[i, "feature type"] == "text":
                shapley_top_10.loc[i, "Features"] = '"' + i + '"'
        else:
                shapley_top_10.loc[i, "Features"] = i + " = " + str(shapley_top_10.loc[i, "feature value"])

    # plot color
    shapley_top_10["positive"] =  shapley_top_10[var]>0
    shapley_top_10["color"] = shapley_top_10.positive.map({True:'tomato', False:'cornflowerblue'}).values.tolist()

    # final plot
    sns.set_style('darkgrid',{'grid.color':".6", "grid.linestyle":":"})
    plt.figure(figsize = (6,4))
    sns.set(font_scale = 1.4)
    sns.set_color_codes("bright")
    plt.title("Feature contributions")
    plot = sns.barplot(x = var,
                      y = "Features",
                      data = shapley_top_10,
                      color = "cornflowerblue",
                      palette = shapley_top_10["color"].values.tolist()
                      )

    # plot annotation
    for i in range(len(shapley_top_10)):
      if shapley_top_10[var].iloc[i] >=0:
          plot.annotate(str(round(shapley_top_10[var].iloc[i],3)), xy =(0,0.2+i), fontsize = 14)
      else:
          plot.annotate(str(round(shapley_top_10[var].iloc[i],3)) , xy =(0,0.2+i), fontsize = 14)

    plt.show()

def shapley_correl_intersec(shapley1, shapley2):
    """ compute correlations between 2 sets of Shapley values, compute intersection for top 3 positive contributions"""
    correl, pval = pearsonr(shapley1, shapley2)
    intersec = len(set(shapley1.sort_values()[-3:].index) & set(shapley2.sort_values()[-3:].index))
    return round(correl,2), pval, intersec

def shapley_sparsity_plus(shapley):
    """ compute sparsity of positive contributions for Shapley values: top 3 positive versus total positive"""
    contribution_plus = shapley[shapley>=0].sort_values(ascending = False)
    sparsity_plus = contribution_plus[:3].sum()/contribution_plus.sum()
    return round(sparsity_plus,2)

def compute_Shapley(instance_to_explain,
                    base_dataset,
                    model,
                    value_function,
                    cat_vars,
                    num_vars,
                    token_var,
                    words,
                    M,
                    T,
                    conv_eps,
                    seed,
                    device, 
                    display):

      """Sampling-based approximation of Shapley values"""

      # result and storage dataframes
      shapley_result_prev = pd.DataFrame() # shapley values (memory for each iteration)
      jsd_shapley_result = dict() # shapley values (jsd)
      uc_shapley_result = dict() # shapley values (uncertainty clf)
      hard_shapley_result = dict() # shapley values (hard)
      perturb_df = pd.DataFrame() # used to store pertubations x_plus and x_minus
      iter_perturb = 0 # index for perturb_df

      # input with only model features
      x = instance_to_explain[cat_vars + num_vars + [token_var]]
      y = instance_to_explain['Y']

      # number of tabular and numerical features
      n_features_tab = len(cat_vars + num_vars)
      n_features_cat = len(cat_vars)
      n_features_num = len(num_vars)

      # seed (for reproducibility and variety of sampling)
      np.random.seed(seed)
      seed_list = np.random.randint(0,len(base_dataset),2*M*T*(n_features_tab + len(instance_to_explain[token_var])))

      for iter_conv in range(1, M+1):

          # features
          feature_idxs_tab = list(range(n_features_tab))  # we use indices for tabular features
          if 0 not in x[n_features_tab].tolist():
              pad_id = x[n_features_tab].shape[0]
          else:
              pad_id = x[n_features_tab].tolist().index(0) # for the token index, we search for the first [PAD] (index should stop there)
          feature_idxs_txt = list(range(pad_id)) # we use indices for text features (tokens)

          shapley_result = pd.DataFrame() # shapley values

          # original feature names, used as columns in perturb_df
          token_var_list = ["token_"+str(i) for i in range(x[n_features_tab].shape[0])][:pad_id]# token list without id 0

          # used to count redundant tokens
          token_count = dict() # key is token and value is its count in the text
          word_list = []


          # compute contribution of each feature
          for i, j in enumerate(feature_idxs_tab + feature_idxs_txt):

              # reinitialize feature lists
              feature_idxs_tab = list(range(n_features_tab))  # we use indices for tabular features
              if 0 not in x[n_features_tab].tolist():
                  pad_id = x[n_features_tab].shape[0]
              else:
                  pad_id = x[n_features_tab].tolist().index(0) # for the token index, we search for the first [PAD] (index should stop there)
              feature_idxs_txt = list(range(pad_id)) # we use indices for text features (tokens)

              # feature to study is tabular
              if i < n_features_tab:
                j_tab = [j]; j_txt = []
                feature_idxs_tab.remove(j) # remove feature j
                feature_name = instance_to_explain.index[j]
                feature_value = instance_to_explain.values[j]
                feature_type = "tabular"

              # feature to study is textual
              else:
                j_tab = []; j_txt = [j]
                feature_idxs_txt.remove(j) # remove token j
                token_id = x[n_features_tab][j]# find corresponding token ID
                feature_name = words[token_id] # corresponding word
                if feature_name in token_count:
                    token_count[feature_name]+=1
                    feature_name = feature_name+"_"+str(token_count[feature_name])
                else:
                    token_count[feature_name]=1
                word_list.append(feature_name)

                feature_value = 1. # presence of word
                feature_type = "text"

              # store feature contribution
              hard_marginal_contributions = []
              jsd_marginal_contributions = []
              uc_marginal_contributions = []

              # Shapley computation via Monte Carlo iterations
              for iter in range(T):

                  # draw random sample z from the base dataset
                  z_full = base_dataset.sample(1, random_state = seed_list[iter_perturb])
                  z = z_full[cat_vars + num_vars + [token_var]].values[0]

                  # 1. tabular case
                  # pick a random subset of features
                  random.seed(seed_list[iter_perturb])
                  x_idx_tab = random.sample(feature_idxs_tab, random.randint(0, len(feature_idxs_tab)))
                  z_idx_tab = [idx for idx in feature_idxs_tab if idx not in x_idx_tab]

                  # construct two new instances
                  x_plus_j_tab = np.array([x[i] if i in x_idx_tab+j_tab else z[i] for i in range(n_features_tab)])
                  x_minus_j_tab = np.array([z[i] if i in z_idx_tab+j_tab else x[i] for i in range(n_features_tab)]) #

                  # 2. text case
                  # pick a random subset of tokens
                  random.seed(seed_list[iter_perturb])
                  n_features_txt = len(feature_idxs_txt)
                  x_idx_txt = random.sample(feature_idxs_txt, random.randint(0, len(feature_idxs_txt)))
                  z_idx_txt = [idx for idx in feature_idxs_txt if idx not in x_idx_txt]

                  # construct two new instances
                  # a) where all tokens in x with index in z_idx_txt are replaced by 0 if the token ID is not in z
                  x_plus_j_txt = x[n_features_tab].copy()
                  for txt_idx in range(pad_id):
                      if txt_idx in z_idx_txt:
                          if x[n_features_tab][txt_idx] not in z[n_features_tab]:
                            x_plus_j_txt[txt_idx] = 0


                  # b) where all tokens in x with index in z_idx_txt + j_txt are replaced by 0 if the token ID is not in z
                  x_minus_j_txt = x[n_features_tab].copy()
                  for txt_idx in range(pad_id):
                      if txt_idx in z_idx_txt+j_txt:
                          if x[n_features_tab][txt_idx] not in z[n_features_tab]:
                            x_minus_j_txt[txt_idx] = 0

                  ## compute x+ and x- (perturbations)
                  # x_plus:convert to tensor and loader
                  text_plus = torch.tensor([x_plus_j_txt])
                  categorical_plus = torch.tensor([x_plus_j_tab[:-n_features_num]]).int()
                  numerical_plus = torch.tensor([x_plus_j_tab[-n_features_num:]]).float()
                  label = torch.tensor([y]).int()
                  x_plus_tensor = TensorDataset(text_plus, categorical_plus, numerical_plus, label)
                  x_plus_loader = DataLoader(x_plus_tensor, batch_size = 1)
                  #  x_minus: convert to tensor and loader
                  text_minus = torch.tensor([x_minus_j_txt])
                  categorical_minus = torch.tensor([x_minus_j_tab[:-n_features_num]]).int()
                  numerical_minus = torch.tensor([x_minus_j_tab[-n_features_num:]]).float()
                  label = torch.tensor([y]).int()
                  x_minus_tensor = TensorDataset(text_minus, categorical_minus, numerical_minus, label)
                  x_minus_loader = DataLoader(x_minus_tensor, batch_size = 1)

                  # compute softmax and jsd
                  text_preds_plus, text_softs_plus, tabular_preds_plus, tabular_softs_plus, jsd_plus, preds_plus = value_function_jsd(x_plus_loader, model, model_type = "TTT", seed = seed, device = device)
                  softs_plus = np.concatenate((text_softs_plus,tabular_softs_plus), axis=1)
                  disagreement_plus = int((tabular_preds_plus!=text_preds_plus)[0])
                  text_preds_minus, text_softs_minus, tabular_preds_minus, tabular_softs_minus, jsd_minus, preds_minus = value_function_jsd(x_minus_loader, model, model_type = "TTT", seed = seed, device = device)
                  softs_minus = np.concatenate((text_softs_minus,tabular_softs_minus), axis=1)
                  disagreement_minus = int((tabular_preds_minus!=text_preds_minus)[0])

                  # compute contribution with true uncertainty estimate
                  hard_marginal_contributions.append(disagreement_plus - disagreement_minus)
                  # value_function = jsd
                  jsd_marginal_contributions.append(jsd_plus - jsd_minus)
                  # value_function = uncertainty classifier
                  v_plus = value_function.predict_proba(softs_plus)[:,1][0]
                  v_minus = value_function.predict_proba(softs_minus)[:,1][0]
                  uc_marginal_contributions.append(v_plus - v_minus)

                  # store perturbations
                  perturb_df.loc[iter_perturb,cat_vars + num_vars] = x_plus_j_tab
                  perturb_df.loc[iter_perturb,token_var_list] = x_plus_j_txt[:pad_id]
                  perturb_df.loc[iter_perturb,"jsd"] = jsd_plus
                  perturb_df.loc[iter_perturb,"uncertainty_classifier"] = v_plus
                  perturb_df.loc[iter_perturb,"stream_disagreement"] = disagreement_plus
                  iter_perturb+=1

                  perturb_df.loc[iter_perturb,cat_vars + num_vars] = x_minus_j_tab
                  perturb_df.loc[iter_perturb,token_var_list] = x_minus_j_txt[:pad_id]
                  perturb_df.loc[iter_perturb,"jsd"] = jsd_minus
                  perturb_df.loc[iter_perturb,"uncertainty_classifier"] = v_minus
                  perturb_df.loc[iter_perturb,"stream_disagreement"] = disagreement_minus
                  iter_perturb+=1


              # store shapley values
              if feature_name in jsd_shapley_result:
                  jsd_shapley_result[feature_name] = jsd_shapley_result[feature_name] + jsd_marginal_contributions
              else:
                  jsd_shapley_result[feature_name] = jsd_marginal_contributions
              if feature_name in uc_shapley_result:
                  uc_shapley_result[feature_name] = uc_shapley_result[feature_name] + uc_marginal_contributions
              else:
                  uc_shapley_result[feature_name] = uc_marginal_contributions
              if feature_name in hard_shapley_result:
                  hard_shapley_result[feature_name] = hard_shapley_result[feature_name] + hard_marginal_contributions
              else:
                  hard_shapley_result[feature_name] = hard_marginal_contributions

              # compute Shapley value with MC approach: average over the iterations
              jsd_shapley_value = sum(jsd_marginal_contributions)/len(jsd_marginal_contributions)
              uc_shapley_value = sum(uc_marginal_contributions)/len(uc_marginal_contributions)
              hard_shapley_value = sum(hard_marginal_contributions)/len(hard_marginal_contributions)

              # result table
              shapley_result.loc[feature_name, "feature value"] = feature_value
              shapley_result.loc[feature_name, "feature type"] = feature_type
              shapley_result.loc[feature_name, "Shapley value (jsd)"] = jsd_shapley_value
              shapley_result.loc[feature_name, "Shapley value (uncertainty classifier)"] = uc_shapley_value
              shapley_result.loc[feature_name, "Shapley value (hard)"] = hard_shapley_value


          # convergence check on hard Shapley value
          if len(shapley_result_prev)>0: # first round :no need to check for convergence
              shapley_result["Shapley value (jsd)"] = (shapley_result["Shapley value (jsd)"] * T + shapley_result_prev["Shapley value (jsd)"] * (iter_conv*T - T)) / (iter_conv*T)
              shapley_result["Shapley value (uncertainty classifier)"] = (shapley_result["Shapley value (uncertainty classifier)"] * T + shapley_result_prev["Shapley value (uncertainty classifier)"] * (iter_conv*T - T)) / (iter_conv*T)
              shapley_result["Shapley value (hard)"] = (shapley_result["Shapley value (hard)"] * T + shapley_result_prev["Shapley value (hard)"] * (iter_conv*T - T)) / (iter_conv*T)

              shapley_mad_hard = np.mean(np.abs(shapley_result["Shapley value (hard)"] - shapley_result_prev["Shapley value (hard)"]))
              shapley_maxad_hard = np.max(np.abs(shapley_result["Shapley value (hard)"] - shapley_result_prev["Shapley value (hard)"]))
              if display:
                  print("iteration:", iter_conv*T,"-", "shapley (hard) max abs difference:",round(shapley_maxad_hard,3))
              if shapley_maxad_hard < conv_eps:
                  if display:
                      print("break")
                  break
          shapley_result_prev = shapley_result.copy()

      # compute standard deviation of Shapley values
      jsd_std = dict()
      uc_std = dict()
      hard_std = dict()
      for j in jsd_shapley_result.keys():
          jsd_std[j] = np.std(jsd_shapley_result[j])
      for j in uc_shapley_result.keys():
          uc_std[j] = np.std(uc_shapley_result[j])
      for j in hard_shapley_result.keys():
          hard_std[j] = np.std(hard_shapley_result[j])

      # plot Shapley value bar charts
      if display:
          plot_shapley(shapley_result, type="jsd")
          plot_shapley(shapley_result, type="uncertainty classifier")
          plot_shapley(shapley_result, type="hard")

      ## format perturb_df
      # set as type int
      perturb_df[cat_vars + token_var_list + ["stream_disagreement"]] = perturb_df[cat_vars + token_var_list + ["stream_disagreement"]].astype(int)
      # rename tokens as words
      mapper1 = dict([(key, value) for i, (key, value) in enumerate(zip(token_var_list, word_list))])
      perturb_df.rename(columns = mapper1, inplace=True)

      ## prepare x_df (original x instance to compare with perturbations)
      x_df = pd.DataFrame(instance_to_explain).T[cat_vars + num_vars + [token_var, "jsd", "uncertainty_classifier"]]
      x_df[word_list] = x[token_var][:pad_id]
      x_df.drop(token_var, axis=1, inplace=True)
      x_df = x_df[cat_vars + num_vars + word_list + ["jsd", "uncertainty_classifier"]]
      x_df["stream_disagreement"] = int(instance_to_explain.text_preds!=instance_to_explain.tab_preds)

      return shapley_result, (jsd_std, uc_std, hard_std), (x_df, perturb_df), iter_conv
      
def shap_kernel(z, p):
    """ Kernel SHAP weights for z unmasked features among p"""
    return (p-1)/(binom(p, z)*(z*(p-z)))

def kernel_shap_consistency(perturb_df, x_df, shapley_result, seed):
    """
    Compute the feature contributions to uncertainty with Kernel SHAP.
    """
    # remove duplicates from the perturbation dataset
    perturb_df.drop_duplicates(inplace=True)
    perturb_df.reset_index(drop=True, inplace=True)
    # repeat original instance
    x_df_repeat = pd.concat([x_df]*len(perturb_df)).reset_index(drop=True)
    # coalition {0,1} dataset
    similar_df = (x_df_repeat.iloc[:,:-3] == perturb_df.iloc[:,:-3]).astype(int)
    # list of features
    feature_list = similar_df.columns.tolist()
    # concatenate with target variables
    kernel_shap_dataset = pd.concat((similar_df,perturb_df[['jsd', 'uncertainty_classifier', 'stream_disagreement']]), axis=1)
    # remove rows with only 0s or only 1s (otherwise shap kernel goes to infinity)
    indices_to_drop = similar_df[(similar_df.sum(axis=1)==0) | (similar_df.sum(axis=1)==len(feature_list))].index
    kernel_shap_dataset.drop(indices_to_drop, inplace=True)
    # compute the kernel weights for each row
    kernel_shap_dataset["z"] = kernel_shap_dataset[feature_list].sum(axis=1)
    kernel_shap_dataset["weight"] = 0
    kernel_shap_dataset["weight"] = kernel_shap_dataset.apply(lambda row: shap_kernel(row["z"], len(feature_list)), axis=1)
    # fit linear regression with Lasso regularization (jsd)
    jsd_lasso = Lasso(alpha=1e-3, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=seed, selection='cyclic')
    jsd_lasso.fit(kernel_shap_dataset[feature_list], kernel_shap_dataset["jsd"], sample_weight=kernel_shap_dataset["weight"])
    shapley_result["Kernel Shap (jsd)"] = jsd_lasso.coef_
    # fit linear regression with Lasso regularization (uncertainty_classifier)
    uc_lasso = Lasso(alpha=1e-3, fit_intercept=True, precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=seed, selection='cyclic')
    uc_lasso.fit(kernel_shap_dataset[feature_list], kernel_shap_dataset["uncertainty_classifier"], sample_weight=kernel_shap_dataset["weight"])
    shapley_result["Kernel Shap (uncertainty classifier)"] = uc_lasso.coef_
    # fit logistic regression (hard)
    hard_logistic = LogisticRegression(penalty='l2', tol=0.0001, C=2.0, fit_intercept=True,random_state=seed, solver='lbfgs', max_iter=1000)
    hard_logistic.fit(kernel_shap_dataset[feature_list], kernel_shap_dataset["stream_disagreement"], sample_weight=kernel_shap_dataset["weight"])
    shapley_result["Kernel Shap (hard)"] = hard_logistic.coef_.flatten()

    return shapley_result     
    
def get_softmax(loader, model, model_type, seed, device):
    """
    Returns softmax (last layer) based on input data and model.
    """
    pred_list = []
    y_list = []
    torch.manual_seed(seed)
    for batch in loader:
      # evaluation mode
      model.eval()

      # extract inputs and labels
      text = batch[0]
      categorical = batch[1]
      numerical = batch[2]
      y = batch[3]

      # key padding mask
      if model_type in ["TTT"]:
            text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 2), dtype=int), text))
            mask = text_with_ones==0

      # put to device
      text = text.to(device)
      mask = mask.to(device)
      categorical = categorical.to(device)
      numerical = numerical.to(device)
      y = y.to(device)

      # prediction and MM representation
      with torch.no_grad():
          pred, text_pred, tabular_pred  = model(text, mask, categorical, numerical.float())

      # add predictions to lists
      pred_list.append(pred)
      y_list.append(y)

    # concatenate and compute softmax
    pred_list = torch.cat(pred_list)
    pred_softs = F.softmax(pred_list, dim=1)
    y_list = torch.cat(y_list)

    return pred_softs.cpu().numpy(), y_list.cpu().numpy()

def conformal_prediction(loader_validation, loader_target, model, model_type, target_coverage, seed, device):
    """
    Compute the interval width and coverage by leveraging conformal prediction.
    Conformal score is 1 the probability of true class (Sadinle et al., 2019). 
    """
    # get softmax and label for validation dataset
    validation_softmax, validation_label = get_softmax(loader_validation, model, model_type, seed, device)

    # compute conformal score s (1-probability of true class) on the validation dataset
    s = 1 - validation_softmax[np.arange(len(validation_softmax)),validation_label]

    # compute the corrected quantile of conformal score
    conformity_quantile = np.quantile(s, q=target_coverage)*(len(s)+1)/len(s)

    # get softmax and label for target dataset
    target_softmax, target_label = get_softmax(loader_target, model, model_type, seed, device)

    # compute mean interval width
    interval_width = ((1-target_softmax)<=conformity_quantile).sum()/len(target_softmax)

    # compute coverage
    coverage = ([target_label[i] for i in np.where((1-target_softmax)<=conformity_quantile)[0]] == np.where((1-target_softmax)<=conformity_quantile)[1]).sum()/len(target_softmax)

    return coverage, interval_width


def compute_MCD(model, loader, n_simu, seed, device):
    """Compute MCD simulations: total and aleatoric uncertainty"""

    # seed (for reproducibility)
    np.random.seed(seed)
    seed_list = np.random.randint(0, 1000, n_simu)

    for i, simu_seed in enumerate(seed_list):

        softmax_list = [] # store softmax
        entropy_list = [] # store entropies
        torch.manual_seed(simu_seed) # set seet

        for text, categorical, numerical, y in loader:

            # training mode (dropout)
            model.train()

            # key padding mask
            text_with_ones = torch.column_stack((torch.ones((y.shape[0],categorical.shape[1]+numerical.shape[1] + 2), dtype=int), text))
            padding_mask = text_with_ones==0

            # to device
            text = text.to(device)
            padding_mask = padding_mask.to(device)
            categorical = categorical.to(device)
            numerical = numerical.to(device)
            y = y.to(device)

            # prediction
            with torch.no_grad():
              pred = model(text, padding_mask, categorical, numerical.float())[0]

            # compute softmax probabilities
            p_hat = F.softmax(pred, dim=1)

            # entropy
            ent = torch.tensor(entropy(p_hat.cpu().numpy(), base=2, axis=1)).to(device)

            # store softmax distributions
            softmax_list.append(p_hat)

            # store entropies
            entropy_list.append(ent)

        if i == 0:
          softmax_list_sum = torch.cat(softmax_list)
          entropy_list_sum = torch.cat(entropy_list)
        else:
          softmax_list_sum += torch.cat(softmax_list)
          entropy_list_sum += torch.cat(entropy_list)

    # compute average over iterations
    softmax_list_avg = softmax_list_sum/(i+1)
    entropy_list_avg = entropy_list_sum/(i+1)

    # compute Shannon entropy
    shannon_entropy = entropy(softmax_list_avg.cpu().numpy(), base=2, axis=1)

    return shannon_entropy.mean(), entropy_list_avg.mean().item()