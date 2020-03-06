import pandas as pd
import numpy as np
import os

import numpy as np
import torch
from torch.distributions.uniform import Uniform
from torch.distributions.binomial import Binomial

if torch.cuda.is_available():
    print("cuda is available")
    import torch.cuda as t
    device = torch.device('cuda:0')
else:
    print("cuda is unavailable")
    import torch as t
    device = torch.device('cpu')


data_path = "../Data/files_needed"

# ------------------- Read in Data -----------------------------------------
# column names
column_names_annotation = pd.read_csv(os.path.join(data_path,
                                                   "column_names_all_MERIT_variants.txt"),
                                      sep="@", header=None)

fname = "chr4_all_MERIT_variants.txt"
chr4_all = pd.read_csv(os.path.join(data_path,fname), sep="\t", names=column_names_annotation[2])
chr4_all.dropna(inplace=True)

# Boolean series: True if total_depth >= 50
enough_tot_read = chr4_all.total_depth_frw + chr4_all.total_depth_rev >= 50
# Select those rows with total_depth >=50
chr4_df = chr4_all[enough_tot_read]

# turn off an unnecessary warning
pd.options.mode.chained_assignment = None
# convert gTUM to B, GL/GLnew to T, and granu to G.
category_series = (chr4_df['ID'].str.split('_').str[-2].str.replace('gTUM','B')
                   .str.replace('GLnew','T').str.replace('GL','T')
                   .str.replace('granu','G') )
# assign the category names to a new column called 'Category'
chr4_df['Category'] = category_series.copy()

# ----------------- pick single nucleotide polymorphysm: C to A ------------------------------
ref = 'C'
alt = 'A'
# alt_condition = (chr4_df['alt_depth_rev'] + chr4_df['alt_depth_frw']) > 1
SNP_df = chr4_df[(chr4_df.ref == ref)&(chr4_df.alt==alt)]
N_array = np.array(SNP_df['total_depth_frw']+SNP_df['total_depth_rev'])
N_ls = t.FloatTensor(N_array)

n_array = np.array(SNP_df['alt_depth_frw']+SNP_df['alt_depth_rev'])
n_ls = t.FloatTensor(n_array)

print('n_ls is on device ', n_ls.get_device())
print('N_ls is on device', N_ls.get_device())

# Split into train and validation sets
S = len(N_ls)
shuffled_indice = torch.randperm(S)
N_ls_shuffled = N_ls[shuffled_indice]
n_ls_shuffled = n_ls[shuffled_indice]

# percentage of train set.
train_frac = 0.7
train_index = int(0.7*S)
N_ls_train = N_ls_shuffled[0:train_index]
N_ls_valid = N_ls_shuffled[train_index:]
n_ls_train = n_ls_shuffled[0:train_index]
n_ls_valid = n_ls_shuffled[train_index:]


from BinomialMixture_object import BinomialMixture
# We only want to train on the training set.
N_ls = N_ls_train
n_ls = n_ls_train


# Sample size
S = len(N_ls)
S_val = len(N_ls_valid)

K_list = range(2, 9)
params_list = []

logL_list = []
AIC_list = []
BIC_list = []

logL_val_list = []
AIC_val_list = []
BIC_val_list = []

# Set K, the number of Binomial distributions in the to-be-fitted mixture model
for K in K_list:

    BM = BinomialMixture(n_components=K, tolerance=1e-5, max_step=int(1e4), verbose=False)
    BM.fit(N_ls, n_ls)
    log_likelihood = BM.calc_logL(N_ls, n_ls)
    AIC, BIC = BM.calc_AIC_BIC(N_ls, n_ls)
    pi_list = BM.pi_list
    theta_list = BM.theta_list
    params = torch.stack([pi_list, theta_list], dim=1)

    #  calculate metrics for the validation sets
    log_likelihood_val = BM.calc_logL(N_ls_valid, n_ls_valid)
    AIC_val = -2.0/float(S_val)*log_likelihood_val + 2.0*(2.0*float(K)+1.0)/float(S_val)
    BIC_val = -2.0*log_likelihood_val + np.log(float(S_val))*(2.0*float(K)+1.0)

    print(f"{params}")
    print(f"Akaike Information Criterion (AIC) = {AIC:.6f}")
    print(f"Bayesian Information Criterion (BIC) = {BIC:.6f}")

    logL_list.append(log_likelihood)
    AIC_list.append(AIC)
    BIC_list.append(BIC)
    params_list.append(params)

    logL_val_list.append(log_likelihood_val)
    AIC_val_list.append(AIC_val)
    BIC_val_list.append(BIC_val)


print(f"K = {K_list}")

print(f"logL = {logL_list}")
print(f"logL_val = {logL_val_list}")

print(f"AIC = {AIC_list}")
print(f"AIC_val = {AIC_val_list}")

print(f"BIC = {BIC_list}")
print(f"BIC_val = {BIC_val_list}")

torch.save(K_list, "K_list.pt")
torch.save(params_list, "params_list.pt")

torch.save(logL_list, "logL_list.pt")
torch.save(AIC_list, "AIC_list.pt")
torch.save(BIC_list, "BIC_list.pt")

torch.save(logL_val_list, "logL_val_list.pt")
torch.save(AIC_val_list, "AIC_val_list.pt")
torch.save(BIC_val_list, "BIC_val_list.pt")
