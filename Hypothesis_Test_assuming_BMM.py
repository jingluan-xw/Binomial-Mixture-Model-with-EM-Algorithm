import pandas as pd
print("pandas imported")
import numpy as np
import os

import numpy as np
import torch
from torch.distributions.uniform import Uniform
from torch.distributions.binomial import Binomial
from BinomialMixture_object import BinomialMixture

if torch.cuda.is_available():
    print("cuda is available")
    import torch.cuda as t
    device = torch.device('cuda:0')
else:
    print("cuda is unavailable")
    import torch as t
    device = torch.device('cpu')


data_path = "../Data/files_needed"
save_file_path = "../Output_files"

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
# N_Series = pd.Series(SNP_df['total_depth_frw']+SNP_df['total_depth_rev'])
N_array = np.array(SNP_df['total_depth_frw']+SNP_df['total_depth_rev'])
N_ls_all = t.FloatTensor(N_array)

# n_Series = pd.Series(SNP_df['alt_depth_frw']+SNP_df['alt_depth_rev'])
n_array = np.array(SNP_df['alt_depth_frw']+SNP_df['alt_depth_rev'])
n_ls_all = t.FloatTensor(n_array)

# N_Series.to_csv(os.path.join(save_file_path, "N_series_5.csv"), index=False)
# n_Series.to_csv(os.path.join(save_file_path, "n_series_5.csv"), index=False)
# print(f"N_series.head() = {N_Series.head()}")
# print(f"n_series.head() = {n_Series.head()}")


from sklearn.model_selection import KFold
n_parts = 5
KF = KFold(n_splits = n_parts, shuffle=True, random_state=934)

for K in range(1,9):

    print(f"K = {K}")
    params_list = []
    pvalue_list = []
    AIC_list = []
    n_all = []
    N_all = []

    kfold_num = 0
    for train_index, valid_index in KF.split(N_ls_all):
        kfold_num += 1
        # Divide into train and valid sets.
        N_ls, N_ls_valid = N_ls_all[train_index], N_ls_all[valid_index]
        n_ls, n_ls_valid = n_ls_all[train_index], n_ls_all[valid_index]

        # Sample size
        S = len(N_ls)
        S_val = len(N_ls_valid)

        # BinomialMixture object is established
        BM = BinomialMixture(n_components=K, tolerance=1e-5, max_step=int(5e4), verbose=False)

        # fit the BinomialMixture object using training data.
        BM.fit(N_ls, n_ls)

        AIC, BIC = BM.calc_AIC_BIC(N_ls, n_ls)
        AIC_list.append(AIC)

        # Store the fitted parameters.
        pi_list = BM.pi_list
        theta_list = BM.theta_list
        params = torch.stack([pi_list, theta_list], dim=1)
        params_list.append(params)

        #  N_ls_valid and n_ls_valid together
        N_all.append(N_ls_valid)
        n_all.append(n_ls_valid)

        # Calculate the p values for the valid data set
        pvalues = BM.p_value(N_ls_valid, n_ls_valid, side='right')
        # Store the p-values for the valid data set
        pvalue_list.append(pvalues)


    print(f"save files {K}")
    torch.save(t.FloatTensor(AIC_list), "../Output_files/AIC_list_K"+str(K)+".pt")
    torch.save(torch.stack(params_list), "../Output_files/params_list_K"+str(K)+".pt")

    # first concat then save
    N_all_concat = torch.cat(N_all, dim=0)
    n_all_concat = torch.cat(n_all, dim=0)
    pvalue_concat = torch.cat(pvalue_list, dim=0)
    # save them
    torch.save(N_all_concat, "../Output_files/N_concat_K"+str(K)+".pt")
    torch.save(n_all_concat, "../Output_files/n_concat_K"+str(K)+".pt")
    torch.save(pvalue_concat, "../Output_files/pvalue_concat_K"+str(K)+".pt")
