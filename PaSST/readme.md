# PaSST: Efficient Training of Audio Transformers with Patchout

Parent Repository : [Efficient Training of Audio Transformers with Patchout](https://github.com/kkoutini/PaSST)

Refer to the above repository for base implementation and setup of PaSST.

## Development environment

If you want to use the same environment as in the paper, you can follow the instructions below.

### Setting up the development experiments environment

For training models from scratch or fine-tuning using the same setup as in the paper:

1. If needed, create a new environment with python 3.8 and activate it:

```bash
conda create -n passt python=3.8
conda activate passt
 ```

1. Install pytorch build that suits your system. For example:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

 ```

1. Install the requirements:

 ```bash
pip install -r requirements.txt
 ```

## Installing OPL

[Download](https://github.com/kahnchana/opl) or clone the opl repository in a folder named OPL in PaSST directory.

Run the following command to clone the repository
```bash
git clone https://github.com/kahnchana/opl
```
## Dataset Download and Preparation

Dataset can be downloaded from [Here](https://zenodo.org/records/3819968).
The dataset is in the form of a zip file. Unzip the file and place the folder in the PaSST directory.
Segregate the data in form of audio files and meta files. Inside the audio files segregate into train and test folders.

Use the fold1_train csv file to get the train data and fold1_evaluate to get the validation data. We have used fold1_evaluate as the metric to compute results on test data as labels have not been provided for fold1_test. 
Use the prep.py script on the above csvs to generate csv for final data prep for finetuning.

## Finetuning on Dcase2020 Datset

Enter the correct paths to point to datasets and meta files in /Dcase_2020/dataset.py.
```python
def default_config():
    name = 'Dcase_2020'  # dataset name
    normalize = False  # normalize dataset
    subsample = False  # subsample squares from the dataset
    roll = True  # apply roll augmentation
    fold = 1
    base_dir = "/users/PaSST_data/"  # base directory of the dataset as downloaded
    if LMODE:
        base_dir = "/system/user/publicdata/CP/audioset/audioset_hdf5s/esc50/"
    meta_train_csv = base_dir + r"meta/Dcase_2020_train.csv"
    meta_test_csv = base_dir + r'meta/Dcase2020_val.csv'
    audio_train_path= base_dir + 'Audio_files/train'
    audio_test_path= base_dir + 'Audio_files/test'
    ir_path = base_dir + "irs/"
    num_of_classes = 10
```

Choose the required model from the PaSST models. Select the optimum values for time and frequency patchouts. Run the following command to finetune 

```bash
python ex_dcase.py with models.net.s_patchout_t=5 models.net.s_patchout_f=1  basedataset.fold=1 -p
```

# Changing the depth of transformer to be finetuned

You can modify the depth of transformer to be finetuned by heading over to models/passt.py and change the depth parameter as required. We have used the passt_s_kd_p16_128_486. Head overt to function definitoin of model and modify here:
```python
def passt_s_kd_p16_128_ap486(pretrained=False, **kwargs):
    """ PaSST pre-trained on AudioSet
    """
    print("\n\n Loading PaSST pre-trained on AudioSet (with KD) Patch 16 stride 10 structured patchout mAP=486 \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, **depth=10**, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (10, 10):
        warnings.warn(
            f"This model was pre-trained with strides {(10, 10)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}.")
    model = _create_vision_transformer(
        'passt_s_kd_p16_128_ap486', pretrained=pretrained, distilled=True, **model_kwargs)
    return model
```
## Embedding Extraction

For embedding extraction purposes, we used embeddings generated by hear21passt module. But to generate embeddings for the 12 blocks, you need to make some modifications to the files

After installing hear21passt library, head to hear21passt/model/passt.py file and make the following changes in the PaSST class under forward pass function:
Replace
```python
    x = self.blocks(x)
    if first_RUN: print(f" after {len(self.blocks)} atten blocks x", x.shape)
    x = self.norm(x)
    if self.dist_token is None:
        return self.pre_logits(x[:, 0])
    else:
        return x[:, 0], x[:, 1]
```
with
```python
    temp=x
        att_blocks = []
        for block in self.blocks:
            temp = block(temp)
            temp2 = self.norm(temp)
            att_blocks.append(((temp2[:,0]+temp2[:,1])/2))
        x = self.blocks(x)
        if first_RUN: print(f" after {len(self.blocks)} atten blocks x", x.shape)
        x = self.norm(x)
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:,1],att_blocks
```

## Classification Scripts

Run the scene/location/device classifier scripts to obtain the desrired results. Make sure to replace the path files in the scripts before execution. 

```python
data = pd.read_csv("your_train_embedding_csv_file_path")
test_data = pd.read_csv("your_test__embedding_csv_file_path")
```

## Librosa Version Clash 

Some users may get a librosa version clash while trying to extract embeddings using models/passt.py directly. To resolve this, you can take an alternate route to first generate mel spectograms using librosa in a different enviornment (python 3.12) worked for me, and then generate embeddings using passt models. 

