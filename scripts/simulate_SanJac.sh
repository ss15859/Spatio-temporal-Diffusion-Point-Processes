cd ../
dataset=SanJac
model=DSTPP
seq_len=150
catalog_path=../../Datasets/QTM/SanJac_catalog.csv
Mcut=1.0
auxiliary_start=2008-01-01:00:00:00
train_nll_start=2009-01-01:00:00:00
val_nll_start=2014-01-01:00:00:00
test_nll_start=2016-01-01:00:00:00
test_nll_end=2018-01-01:00:00:00
timesteps=500
samplingsteps=500
batch_size=64
total_epochs=2000
cuda_id=0
marked_output=1

save_path=./ModelSave/dataset_${dataset}_model_${model}_sigma_${sigma_time}_${sigma_loc}_log_${log_normalization}_seed_${seed}_dim_${cond_dim}_lambda_${loss_lambda}_seq_len_${seq_len}_marked_output_${marked_output}/

python app.py --day_number $1 --batch_size $2 --save_path ${save_path} --dataset ${dataset} --mode sample --model ${model} --timesteps ${timesteps}  --samplingsteps ${samplingsteps}  --seq_len ${seq_len} --catalog_path ${catalog_path} --Mcut ${Mcut} --auxiliary_start ${auxiliary_start} --train_nll_start ${train_nll_start} --val_nll_start ${val_nll_start} --test_nll_start ${test_nll_start} --test_nll_end ${test_nll_end} --marked_output ${marked_output} --weight_path ${save_path}model_best.pkl
