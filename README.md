# consistency

# Inconsistency of LLMs

This repo provides code for "Inconsistency of Large Language Models In Molecular Representations" by Bing Yan and Kyunghyun Cho. Note that this repo can be found at https://github.com/bingyan4science/consistency.

## Dependencies

The code has been tested on Python 3.10 and PyTorch 2.1.0.

* Python 3.10.13: `conda create --prefix ../conda_consist python=3.10`
* PyTorch: https://pytorch.org/get-started/locally/ `conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia`

## Controlled experiments: Train a reaction prediction model based on GPT-2 Small

For the controlled experiments, we train a reaction prediction model by fine-tuning the GPT-2 Small model on SMILES vs IUPAC representations for inputs, and equal chance of generating SMILES and IUPAC outputs:

```
# SMILES representation for inputs
export MODE=smiles
export FOLDER=data/llasmol_reaction_${MODE}_combined_80k_noinstruction
export MODEL=gpt2
export EPOCHS=60
export LR=5e-5
export BSZ=16
export SAVE=train_models/combined_gpt2/${MODE}_e${EPOCHS}_onebatch_bsz${BSZ}_lr${LR}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_teacher_combined.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --base_model $MODEL \
    --batch_size $BSZ \
    --save_model $SAVE \
    --generate_onebatch \
    > ${SAVE}/log.train 2>&1

# IUPAC representation for inputs
export MODE=iupac
export FOLDER=data/llasmol_reaction_${MODE}_combined_80k_noinstruction
export MODEL=gpt2
export EPOCHS=60
export LR=5e-5
export BSZ=16
export SAVE=train_models/combined_gpt2/${MODE}_e${EPOCHS}_onebatch_bsz${BSZ}_lr${LR}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_teacher_combined.py \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --epochs $EPOCHS \
    --lr $LR \
    --base_model $MODEL \
    --batch_size $BSZ \
    --save_model $SAVE \
    --generate_onebatch \
    > ${SAVE}/log.train 2>&1
```

Next, we test the trained model.
```
# SMILES representation for inputs
export MODE=smiles
export FOLDER=data/llasmol_reaction_${MODE}_combined_80k_noinstruction
export MODEL=gpt2
export EPOCHS=60
export LR=5e-5
export BSZ=16
export SAVE=train_models/combined_gpt2/${MODE}_e${EPOCHS}_onebatch_bsz${BSZ}_lr${LR}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_teacher_combined_evaluate.py \
    --data_folder ${FOLDER} \
    --epochs $EPOCHS \
    --base_model $MODEL \
    --save_model $SAVE \
    > ${SAVE}/log.gen.cont 2>&1

# IUPAC representation for inputs
export MODE=iupac
export FOLDER=data/llasmol_reaction_${MODE}_combined_80k_noinstruction
export MODEL=gpt2
export EPOCHS=60
export LR=5e-5
export BSZ=16
export SAVE=train_models/combined_gpt2/${MODE}_e${EPOCHS}_onebatch_bsz${BSZ}_lr${LR}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_teacher_combined_evaluate.py \
    --data_folder ${FOLDER} \
    --epochs $EPOCHS \
    --base_model $MODEL \
    --save_model $SAVE \
    > ${SAVE}/log.gen 2>&1
```

Then we evaluate the prediction and calculate the consistency between SMILES and IUPAC inputs.

```
python calc_consistency.py
```

## Probe experiments: Train a linear classifier on the 6th layer hidden states to get functional groups

First, we train and test the probe.

```
# SMILES representation for inputs
export MODE=smiles
export FOLDER=data/llasmol_reaction_${MODE}_combined_80k_noinstruction
export MODEL=gpt2
export EPOCHS=60
export LR=5e-5
export BSZ=16
export E=10
export LAYER=5
export OLDSAVE=train_models/combined_gpt2/${MODE}_e${EPOCHS}_onebatch_bsz${BSZ}_lr${LR}
export SAVE=${OLDSAVE}/fg_e${E}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_teacher_combined_fg.py \
    --from_pretrained  ${OLDSAVE}/checkpoint_${E} \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --test_path ${FOLDER}/test_smiles.txt \
    --epochs 1 \
    --lr $LR \
    --layer $LAYER \
    --base_model $MODEL \
    --batch_size $BSZ \
    --save_model $SAVE \
    > ${SAVE}/log.train.and.gen 2>&1

# IUPAC representation for inputs
export MODE=iupac
export FOLDER=data/llasmol_reaction_${MODE}_combined_80k_noinstruction
export MODEL=gpt2
export EPOCHS=60
export LR=5e-5
export BSZ=16
export E=10
export LAYER=5
export OLDSAVE=train_models/combined_gpt2/${MODE}_e${EPOCHS}_onebatch_bsz${BSZ}_lr${LR}
export SAVE=${OLDSAVE}/fg_e${E}
echo $SAVE
mkdir -p $SAVE
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 stdbuf -oL -eL python src/train_teacher_combined_fg.py \
    --from_pretrained  ${OLDSAVE}/checkpoint_${E} \
    --train_path ${FOLDER}/train.txt \
    --val_path ${FOLDER}/valid.txt \
    --test_path ${FOLDER}/test_smiles.txt \
    --epochs 1 \
    --lr $LR \
    --layer $LAYER \
    --base_model $MODEL \
    --batch_size $BSZ \
    --save_model $SAVE \
    > ${SAVE}/log.train.and.gen 2>&1
```

Then we test if the reaction prediction consistency correlates with the probe's consistency

```
python calc_probe_consistency.py
```

