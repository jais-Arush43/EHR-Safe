import pandas as pd
import torch
from tqdm import tqdm
import os
import gc
from models import CategoricalAutoEncoder, WGANGP, EncoderDecoder,StochasticNormalizer
from utils.data_generator_utils import *
from preprocess.one_hot_encoders import *



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

generator = WGANGP(encoder_state_dim=256,latent_dim=256)
generator.load_checkpoint(filename="weights/best_gan.pt",map_location=device)
sn_dim = 1           
sce_latent_dim = 32 
tn_dim = 25               
tce_latent_dim = 59         
sc_dim = 3
tc_dim = 6                     
decoder = EncoderDecoder(sn_dim=sn_dim,sce_latent_dim=sce_latent_dim,tn_dim=tn_dim,
            tce_latent_dim=tce_latent_dim,sc_dim=sc_dim,tc_dim=tc_dim,latent_dim=256)
decoder.load_checkpoint(filename="weights/best_encoder_decoder_ckpt.pt",map_location=device)
decoder.to(device)
sn_hat, sc_hat, sn_mask_hat, sc_mask_hat, tn_hat, tc_hat, un_hat, uc_hat, tn_mask_hat, tc_mask_hat = generate_synthetic_dataset(generator, decoder, total_samples=70000, batch_size=64, device=None)

normalizer_sn = StochasticNormalizer()
normalizer_sn.load_params("weights/static_params.pt")

normalizer_tn = StochasticNormalizer()
normalizer_tn.load_params("weights/temporal_params.pt")

temporal_columns = ['DATE', 'Body Height', 'Body Mass Index', 'Body Weight', 'Calcium',
                   'Carbon Dioxide', 'Chloride', 'Creatinine',
                   'DXA [T-score] Bone density', 'Diastolic Blood Pressure',
                   'Egg white IgE Ab in Serum', 'Estimated Glomerular Filtration Rate',
                   'FEV1/​FVC', 'Glucose', 'HIV status',
                   'Hemoglobin A1c/Hemoglobin.total in Blood',
                   'High Density Lipoprotein Cholesterol',
                   'Low Density Lipoprotein Cholesterol', 'Microalbumin Creatine Ratio',
                   'Oral temperature', 'Potassium', 'Sodium', 'Systolic Blood Pressure',
                    'Total Cholesterol', 'Triglycerides', 'Urea Nitrogen']

static_df,temporal_df = denormalize_generated_data(sn_hat, tn_hat, un_hat,sn_mask_hat,tn_mask_hat,
                                            normalizer_sn, normalizer_tn,temporal_columns)


os.makedirs("generated", exist_ok=True)

static_df.to_csv('generated/static_numerical.csv',index=False)
temporal_df.to_csv('generated/temporal_numerical.csv',index=False)

static_input_dims = [5, 2, 21]
static_categorical_cols = ["RACE", "GENDER", "ETHNICITY"]
static_model_instance = CategoricalAutoEncoder(static_input_dims, filename="weights/static_categorical_encoder_decoder.pt")
static_model_instance = static_model_instance.to(device)
static_model_instance.load_model(map_location=device)
static_manager = StaticOneHotManager()
static_manager.load("weights/static_encoded.pkl")
_, sc_onehot = static_model_instance.decode(sc_hat, mask=sc_mask_hat, return_onehot=True)

start = 0
onehot_lists = {}
for col_name, dim in zip(static_categorical_cols, static_input_dims):
    end = start + dim
    onehot_lists[col_name] = [sc_onehot[i, start:end].cpu().tolist() for i in range(sc_onehot.size(0))]
    start = end

onehot_df = pd.DataFrame(onehot_lists)
onehot_df.head()

decoded_dict = {}
for col_name in static_categorical_cols:
    decoded_dict[col_name] = static_manager.inverse_transform(onehot_df[col_name], col_name)

decoded_df = pd.DataFrame(decoded_dict)
decoded_df.head()

decoded_df.to_csv('generated/static_categorical.csv',index=False)

temporal_input_dims = [76, 68, 126, 28, 99, 80]
temporal_categorical_cols = ['CAREPLAN', 'REASON', 'CONDITIONS', 'ENCOUNTER_TYPE',
       'MEDICINE', 'PROCEDURES']


temporal_model_instance = CategoricalAutoEncoder(temporal_input_dims, filename="weights/temporal_categorical_encoder_decoder.pt")
temporal_model_instance = temporal_model_instance.to(device)
temporal_model_instance.load_model(map_location=device)

temporal_manager = TemporalOneHotManager()
temporal_manager.load("weights/temporal_encoded.pkl")

tc_hat_tensor = torch.cat(tc_hat, dim=0).to(device)
tc_mask_hat_tensor = torch.cat(tc_mask_hat, dim=0).to(device)
preds, recon = temporal_model_instance.decode(tc_hat_tensor, mask=tc_mask_hat_tensor, return_onehot=True)




BATCH_SIZE = 1024
SAVE_DIR = "onehot_parts"
os.makedirs(SAVE_DIR, exist_ok=True)

for batch_idx, chunk in enumerate(tqdm(torch.split(recon, BATCH_SIZE), desc="Processing batches")):
    start = 0
    batch_dict = {}
    for col_name, dim in zip(temporal_categorical_cols, temporal_input_dims):
        end = start + dim
        col_array = chunk[:, start:end].numpy()
        col_int = col_array.argmax(axis=1)
        batch_dict[col_name] = col_int
        start = end
    batch_df = pd.DataFrame(batch_dict)
    batch_df.to_parquet(f"{SAVE_DIR}/onehot_batch_{batch_idx:04d}.parquet", index=False)
    del batch_df, batch_dict, chunk, col_array, col_int
    gc.collect()

import glob
files = sorted(glob.glob(f"{SAVE_DIR}/onehot_batch_*.parquet"))
onehot_df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
print("Done. One-hot DataFrame shape:", onehot_df.shape)

decoded_temporal = pd.DataFrame()
for col in temporal_categorical_cols:
    decoded_temporal[col] = temporal_manager.inverse_transform2(onehot_df[col], col)
decoded_temporal.head()



normalizer_uc = StochasticNormalizer()
normalizer_uc.load_params("weights/categorical_times_params.pt")
uc_time = denormalize_uc(uc_hat,normalizer_uc)

decoded_temporal.insert(0, "DATE", uc_time.cpu().numpy())

decoded_temporal.to_csv('generated/temporal_categorical.csv',index=False)