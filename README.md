# TM2T: Stochastical and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts (ECCV 2022)
## [[Project Page]](https://ericguo5513.github.io/TM2T) [[Paper]](https://arxiv.org/abs/2207.01696.pdf)

![teaser_image](https://github.com/EricGuo5513/TM2T/blob/main/docs/teaser_image.png)
  
## Python Virtual Environment

Anaconda is recommended to create this virtual environment.
  
  ```sh
  conda create -f environment.yaml
  conda activate tm2t
  ```
  
If you cannot successfully create the environment, here is a list of required libraries:
  ```
  Python = 3.7.9   # Other version may also work but is not tested.
  PyTorch = 1.6.0 (conda install pytorch==1.6.0 torchvision==0.7.0 -c pytorch)  #Other version may also work but are not tested.
  scipy
  numpy
  tensorflow       # For use of tensorboard only
  spacy
  tqdm
  ffmpeg = 4.3.1   # Other version may also work but are not tested.
  matplotlib = 3.3.1
  nlpeval (https://github.com/Maluuba/nlg-eval)     # For evaluation of motion-to-text only
  bertscore (https://github.com/Tiiiger/bert_score) # For evaluation of motion-to-text only
  ```
  
  After all, if you want to generate 3D motions from customized raw texts, you still need to install the language model for spacy. 
  ```sh
  python -m spacy download en_core_web_sm
  ```
  
  ## Download Data & Pre-trained Models
  
  **If you just want to play our pre-trained models, you don't need to download datasets.**
  ### Datasets
  We are using two 3D human motion-language dataset: HumanML3D and KIT-ML. For both datasets, you could find the details as well as download link [[here]](https://github.com/EricGuo5513/HumanML3D).   
  Please note you don't need to clone that git repository, since all related codes have already been included in current git project.
  
  Download and unzip the dataset files -> Create a dataset folder -> Place related data files in dataset folder:
  ```sh
  mkdir ./dataset/
  ```
  Take HumanML3D for an example, the file directory should look like this:  
  ```
  ./dataset/
  ./dataset/HumanML3D/
  ./dataset/HumanML3D/new_joint_vecs/
  ./dataset/HumanML3D/texts/
  ./dataset/HumanML3D/Mean.mpy
  ./dataset/HumanML3D/Std.npy
  ./dataset/HumanML3D/test.txt
  ./dataset/HumanML3D/train.txt
  ./dataset/HumanML3D/train_val.txt
  ./dataset/HumanML3D/val.txt  
  ./dataset/HumanML3D/all.txt 
  ```
 ### Pre-trained Models
  Create a checkpoint folder to place pre-traine models:
  ```sh
  mkdir ./checkpoints
  ```
    
 #### Download models for HumanML3D from [[here]](https://drive.google.com/file/d/1o7RTDQcToJjTm9_mNWTyzvZvjTWpZfug/view?usp=sharing). Unzip and place them under checkpoint directory, which should be like
```
./checkpoints/t2m/
./checkpoints/t2m/Comp_v6_KLD01/                   # A dumb folder containing information for evaluation dataloading
./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/  # Motion discretizer
./checkpoints/t2m/M2T_EL4_DL4_NH8_PS/              # Motion (token)-to-Text translation model
./checkpoints/t2m/T2M_Seq2Seq_NML1_Ear_SME0_N/     # Text-to-Motion (token) generation model
./checkpoints/t2m/text_mot_match/                  # Motion & Text feature extractors for evaluation
 ```
 #### Download models for KIT-ML [[here]](https://drive.google.com/file/d/1xEoMy1aBRe0fxYeSzeLwzjHr9Ia6d6Gf/view?usp=sharing). Unzip and place them under checkpoint directory.
    
 ## Training Models
 
 All intermediate meta files/animations/models will be saved to checkpoint directory under the folder specified by argument "--name".
 ### Training motion discretizer 
 #### HumanML3D
```sh
python train_vq_tokenizer_v3.py --gpu_id 0 --name VQVAEV3_CB1024_CMT_H1024_NRES3 --dataset_name t2m --n_resblk 3
```
#### KIT-ML
```sh
python train_vq_tokenizer_v3.py --gpu_id 0 --name VQVAEV3_CB1024_CMT_H1024_NRES3 --dataset_name kit --n_resblk 3
```
### Tokenizing all motion data for the following training
#### HumanML3D
```sh
python tokenize_script.py --gpu_id 0 --name VQVAEV3_CB1024_CMT_H1024_NRES3 --dataset_name t2m
```

#### KIT-ML
```sh
python tokenize_script.py --gpu_id 0 --name VQVAEV3_CB1024_CMT_H1024_NRES3 --dataset_name kit
```

### Training motion2text model:
#### HumanML3D
```sh
python train_m2t_transformer.py --gpu_id 0 --name M2T_EL4_DL4_NH8_PS --n_enc_layers 4 --n_dec_layers 4 --proj_share_weight --dataset_name t2m
```
#### KIT-ML
```sh
python train_m2t_transformer.py --gpu_id 0 --name M2T_EL3_DL3_NH8_PS --n_enc_layers 3 --n_dec_layers 3 --proj_share_weight --dataset_name kit
```
### Training text2motion model:
#### HumanML3D
```sh
python train_t2m_joint_seq2seq.py --gpu_id 0 --name T2M_Seq2Seq_NML1_Ear_SME0_N --start_m2t_ep 0 --dataset_name t2m
```
#### KIT-ML
```sh
python train_t2m_joint_seq2seq.py --gpu_id 0 --name T2M_Seq2Seq_NML1_Ear_SME0_N --start_m2t_ep 0 --dataset_name kit
```
### Motion & text feature extractors:
We use the same extractors provided by https://github.com/EricGuo5513/text-to-motion

    
## Generating and Animating 3D Motions (HumanML3D)
#### Translating motions into langauge (using test sets)
With Beam Search:
```sh
python evaluate_m2t_transformer.py --name M2T_EL4_DL4_NH8_PS --gpu_id 2 --num_results 20 --n_enc_rs 4 --n_dec_layers 4 --proj_share_weight --ext beam_search
```

With Sampling:
```sh
python evaluate_m2t_transformer.py --name M2T_EL4_DL4_NH8_PS --gpu_id 2 --num_results 20 --n_enc_layers 4 --n_dec_layers 4 --proj_share_weight --sample --top_k 3 --ext top_3
```

#### Generating motions from texts (using test sets)
```sh
python evaluate_t2m_seq2seq.py --name T2M_Seq2Seq_NML1_Ear_SME0_N --num_results 10 --repeat_times 3 --sample --ext sample
```
where  *--repeat_time* gives how many sampling rounds are carried out for each description. This script will results in 3x10 animations under directory *./eval_results/t2m/T2M_Seq2Seq_NML1_Ear_SME0_N/sample/*.

#### Sampling results from customized descriptions
```sh
python gen_script_t2m_seq2seq.py --name T2M_Seq2Seq_NML1_Ear_SME0_N  --repeat_times 3 --sample --ext customized --text_file ./input.txt
```
This will generate 3 animated motions for each description given in text_file *./input.txt*.

If you find problem with installing ffmpeg, you may not be able to animate 3d results in mp4. Try gif instead.

## Quantitative Evaluations
### Evaluating Motion2Text
```sh
python final_evaluation_m2t.py 
```
### Evaluating Motion2Text
```sh
python final_evaluation_t2m.py 
```
This will evaluate the model performance on HumanML3D dataset by default. You could also run on KIT-ML dataset by uncommenting certain lines in *./final_evaluation.py*. The statistical results will saved to *./m2t(t2m)_evaluation.log*.

### Misc
 Contact Chuan Guo at cguo2@ualberta.ca for any questions or comments.
