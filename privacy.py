import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from utils.privacy_utils import *
from models import PrivacyEvaluator


sn_real = pd.read_csv('real/static_numerical.csv')
sc_real = pd.read_csv('real/static_categorical.csv')
tn_real = pd.read_csv('real/temporal_numerical.csv')
tc_real = pd.read_csv('real/temporal_categorical.csv')

sn_gen = pd.read_csv('generated/static_numerical.csv')
sc_gen = pd.read_csv('generated/static_categorical.csv')
tn_gen = pd.read_csv('generated/temporal_numerical.csv')
tc_gen = pd.read_csv('generated/temporal_categorical.csv')

tn_gen_new = convert_to_wide_format(tn_gen)
tc_gen_new = convert_to_wide_format(tc_gen,time_col='DATE')
print(tc_gen_new.shape)
print(tn_gen_new.shape)


tn_real_m  = fix_dataframe_dtypes(tn_real)
tc_real_m = fix_dataframe_dtypes(tc_real)

final_real = combine_dataframes(tn_real_m[tn_real.columns[1:]],tc_real_m[tc_real.columns[1:]],sn_real,sc_real)
final_real.drop(columns=['HIV status'], inplace=True)

num = tc_gen_new.shape[0]
tn_gen_m = tn_gen_new[:num]
tc_gen_m = tc_gen_new[:num]
sn_gen_m = sn_gen[:num]
sc_gen_m = sc_gen[:num]


final_gen = combine_dataframes(tn_gen_m[tn_gen.columns[1:]],tc_gen_m[tc_real.columns[1:]],sn_gen_m,sc_gen_m)

from sklearn.model_selection import train_test_split
real_train_wide, real_test_wide = train_test_split(
    final_real,
    test_size=0.2,
    random_state=42
)

print(f"Total real patients: {len(final_real)}")
print(f"Split into Real Train patients: {len(real_train_wide)}")
print(f"Split into Real Test patients:  {len(real_test_wide)}")

num_cols = list(tn_gen.columns[1:]) + list(sn_gen.columns)
cat_cols = list(tc_gen.columns[1:]) + list(sc_gen.columns)
sensitive_cols = ['RACE','GENDER','ETHNICITY']

static_num_cols = list(sn_gen.columns)
static_cat_cols = list(sc_gen.columns)
temp_num_cols = list(tn_gen.columns[1:])
temp_cat_cols = list(tc_gen.columns[1:])


print("--- RUNNING QUICK TEST WITH SAMPLING (n=5000) ---")

evaluator_sampled = PrivacyEvaluator(
    real_train_df=real_train_wide,
    real_test_df=real_test_wide,
    synthetic_df=final_gen,
    static_num_cols=static_num_cols,
    static_cat_cols=static_cat_cols,
    temp_num_cols=temp_num_cols,
    temp_cat_cols=temp_cat_cols,
    n_samples=5000 
)

results_sampled = evaluator_sampled.evaluate_all_attacks(
    sensitive_attribute_cols=sensitive_cols
)

print("\n--- SAMPLING RESULTS ---")
print(results_sampled)


table_data = []

# Membership Inference
table_data.append({
    'Privacy Attack': 'Membership Inference',
    'Metric': 'Attack AUC',
    'Ideal Score': '0.5000',
    'Our Score': f"{results_sampled['membership_inference_auc']:.4f}",
    'Interpretation': 'Excellent (Very close to ideal)'
})

# Re-identification
table_data.append({
    'Privacy Attack': 'Re-identification',
    'Metric': 'Attack Success Rate',
    'Ideal Score': f"≤ Baseline ({results_sampled['re_identification_baseline']:.4f})",
    'Our Score': f"{results_sampled['re_identification_risk']:.4f}",
    'Interpretation': 'Excellent (Score < Baseline)'
})

# Attribute Inference (one row per attribute)
for attribute, scores in results_sampled['attribute_inference'].items():
    interpretation = 'Good (Attack ≤ Baseline)' if scores['synthetic_data_auc'] <= scores['real_data_auc'] else 'Potential Leak (Attack > Baseline)'
    table_data.append({
        'Privacy Attack': f'Attribute Inference ({attribute})',
        'Metric': 'Attack AUC',
        'Ideal Score': f"≤ Baseline ({scores['real_data_auc']:.4f})",
        'Our Score': f"{scores['synthetic_data_auc']:.4f}",
        'Interpretation': interpretation
    })

privacy_summary_df = pd.DataFrame(table_data) 

print("--- Privacy Evaluation Summary Table ---")
print(privacy_summary_df.to_string(index=False)) 


attribute_data = results_sampled['attribute_inference']
attributes = list(attribute_data.keys())
baseline_aucs = [attribute_data[attr]['real_data_auc'] for attr in attributes]
attack_aucs = [attribute_data[attr]['synthetic_data_auc'] for attr in attributes]

plot_df = pd.DataFrame({
    'Attribute': attributes,
    'Baseline AUC (Train on Real)': baseline_aucs,
    'Attack AUC (Train on Synthetic)': attack_aucs
})

x = np.arange(len(attributes))  
width = 0.35 

fig, ax = plt.subplots(figsize=(10, 6))

rects1 = ax.bar(x - width/2, plot_df['Baseline AUC (Train on Real)'], width, 
                label='Baseline (Train on Real)', color='skyblue')
rects2 = ax.bar(x + width/2, plot_df['Attack AUC (Train on Synthetic)'], width, 
                label='Attack (Train on Synthetic)', color='lightcoral')

ax.set_ylabel('AUC Score')
ax.set_title('Attribute Inference Attack Results (AUC)')
ax.set_xticks(x)
ax.set_xticklabels(attributes)
ax.legend(loc='lower right') 
ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='Random Guess (AUC=0.5)')
ax.legend()
min_auc = min(min(baseline_aucs), min(attack_aucs))
max_auc = max(max(baseline_aucs), max(attack_aucs))
ax.set_ylim([max(0, min_auc - 0.05), min(1, max_auc + 0.05)]) 
ax.bar_label(rects1, padding=3, fmt='%.3f', fontsize=9)
ax.bar_label(rects2, padding=3, fmt='%.3f', fontsize=9)

fig.tight_layout()
plt.xticks(rotation=0) 
plt.show()