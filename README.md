# DeepMPSF
The source code, training and test datasets of paper 'DeepMPSF: A Deep Learning Network for Predicting General Protein Phosphorylation Sites Based on Multiple Protein Sequence Features'  
## Install dependency:  
* conda create -n phros python == 3.8
* conda activate phros
* pip install --no-cache-dir torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
* pip install pandas
* pip install scikit-learn
## Prerequisites:
* `python`: 3.8
* `CUDA`: 10.1
* `pytorch`: -1.7.0
## All relevant inputs：
Due to the limitation of Github, some inputs larger than 25MB are not uploaded. Please contact me directly at 20215227108@stu.suda.edu.cn.
## Reproduce experimental results:
The prediction of phosphorylation sites by DeepMPSF includes the prediction of S/T and Y. Therefore, there are corresponding datasets and features for S/T and Y, respectively. Here is the example of S/T. And for the prediction or training of Y, you can replace '_ST' with '_Y' in My_Model.py and the corresponding training/prediction script.
### Dataset
D_human:Table_All_train_general_ST.csv/Table_All_train_general_Y.csv  
T_human:Table_All_test_general_ST.csv/Table_All_test_general_Y.csv  
T_homo:Table_blind_human_ST_2k.csv/Table_blind_human_Y_2k.csv
T_mus:Table_blind_musculus_ST_2k.csv/Table_blind_musculus_Y_2k.csv  
T_rattus:Table_blind_Rattus_ST_2k.csv/Table_blind_Rattus_Y_2k.csv
### Test
Step 1: ```cd code```
Step 2: ```python predic_my_model.py```
### Test
Step 1: ```cd code```
Step 2: ```python predic_my_model_blind_homo.py```
### Test
Step 1: ```cd code```
Step 2: ```python predic_my_model_blind_musculus.py```
### Test
Step 1: ```cd code```
Step 2: ```python predic_my_model_blind_Rattus.py```
### Train your own model DeepMPSF
Step 1: ```cd code```
Step 2: ```python train_my_model.py```
