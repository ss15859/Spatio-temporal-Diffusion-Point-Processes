cd ../
dataset=SCEDC
model=DSTPP
seq_len=150
catalog_path=../../Datasets/SCEDC/SCEDC_catalog.csv
Mcut=2.0
auxiliary_start=1981-01-01:00:00:00
train_nll_start=1985-01-01:00:00:00
val_nll_start=2005-01-01:00:00:00
test_nll_start=2014-01-01:00:00:00
test_nll_end=2020-01-01:00:00:00
timesteps=500
samplingsteps=500
batch_size=64
total_epochs=2000
cuda_id=0
marked_output=1

save_path=./ModelSave/dataset_${dataset}_model_${model}_sigma_${sigma_time}_${sigma_loc}_log_${log_normalization}_seed_${seed}_dim_${cond_dim}_lambda_${loss_lambda}_seq_len_${seq_len}_marked_output_${marked_output}/

python app.py --dataset ${dataset} --mode train --save_path ${save_path} --timesteps ${timesteps} --samplingsteps ${samplingsteps} --batch_size ${batch_size} --cuda_id ${cuda_id} --total_epochs ${total_epochs} --seq_len ${seq_len} --catalog_path ${catalog_path} --Mcut ${Mcut} --auxiliary_start ${auxiliary_start} --train_nll_start ${train_nll_start} --val_nll_start ${val_nll_start} --test_nll_start ${test_nll_start} --test_nll_end ${test_nll_end} --marked_output ${marked_output}
