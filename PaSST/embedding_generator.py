import torch 
from hear21passt.base import get_basic_model,get_model_passt
from Mischallaneous import generate_embeddings_and_save_to_csv, extract_info
import pandas as pd
import os
mode='test' #train or test (depending on which embedding is being extracted for)

model1 = get_basic_model(mode="embed_only", arch="passt_s_kd_p16_128_ap486")
model1.net = get_model_passt(arch="passt_s_kd_p16_128_ap486",  n_classes=10, s_patchout_f = 0, s_patchout_t = 0 ,fstride = 10, tstride = 10)

'''If you are using a finetuned version of the above model than use the below script to load the weights after finetuning . If you only want embeddings from the pretrained version , then leave this portion commented out'''

# checkpoint = torch.load(r'lightning_logs/latest_version/checkpoints/ckpt_file', map_location=torch.device('cuda'))
# checkpoint_dict = checkpoint['state_dict'] 
# model1.load_state_dict(checkpoint_dict,strict=False)

if(mode=='train'):
    folder_path='Train audio file path '
if(mode=='test' or mode=='validation'):
    folder_path='Test audio file path'
model=model1
device='cuda'
save_path=generate_embeddings_and_save_to_csv(folder_path, model, device)

df_new=pd.read_csv(save_path)

df_new[['scene', 'location', 'device']] = df_new['file_name'].apply(extract_info).apply(pd.Series)

df_new.to_csv(fr'Classification/Embedding_files/PaSST_Embeddings/finalhear21_embeddings_blocks.csv', index=True)

os.remove(save_path)