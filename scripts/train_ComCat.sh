cd ../
dataset=ComCat
model=DSTPP
seq_len=150
catalog_path=../../Datasets/ComCat/ComCat_catalog.csv
Mcut=2.5
auxiliary_start=1971-01-01:00:00:00
train_nll_start=1981-01-01:00:00:00
val_nll_start=1998-01-01:00:00:00
test_nll_start=2007-01-01:00:00:00
test_nll_end=2020-01-17:00:00:00
timesteps=500
samplingsteps=500
batch_size=128
total_epochs=2000
cuda_id=0
marked_output=1



save_path=./ModelSave/dataset_${dataset}_model_${model}_sigma_${sigma_time}_${sigma_loc}_log_${log_normalization}_seed_${seed}_dim_${cond_dim}_lambda_${loss_lambda}_seq_len_${seq_len}_marked_output_${marked_output}/
# python app.py --loss_lambda ${loss_lambda} --dim 3 --cond_dim ${cond_dim} --save_path ${save_path} --seed ${seed} --log_normalization ${log_normalization} --dataset ${dataset} --mode train --model ${model} --samplingsteps -1 --batch_size 32 --total_epochs 1000 --n_samples 1 --per_step 1 --sigma_time ${sigma_time} --sigma_loc ${sigma_loc} --seq_len ${seq_len} --catalog_path ${catalog_path} --Mcut ${Mcut} --auxiliary_start ${auxiliary_start} --train_nll_start ${train_nll_start} --val_nll_start ${val_nll_start} --test_nll_start ${test_nll_start} --test_nll_end ${test_nll_end} --marked_output ${marked_output}
# python app.py --loss_lambda ${loss_lambda} --cond_dim ${cond_dim} --save_path ${save_path} --seed ${seed} --dim 3 --log_normalization ${log_normalization} --dataset ${dataset} --mode test --model ${model} --langevin_step ${langevin_step} --samplingsteps ${samplingsteps} --n_samples 30 --per_step 250  --sigma_time ${sigma_time} --sigma_loc ${sigma_loc} --seq_len ${seq_len} --catalog_path ${catalog_path} --Mcut ${Mcut} --auxiliary_start ${auxiliary_start} --train_nll_start ${train_nll_start} --val_nll_start ${val_nll_start} --test_nll_start ${test_nll_start} --test_nll_end ${test_nll_end} --marked_output ${marked_output} --weight_path ${save_path}model_best.pkl
# python app.py --day_number $1 --batch_size $2 --loss_lambda ${loss_lambda} --cond_dim ${cond_dim} --save_path ${save_path} --seed ${seed} --dim 3 --log_normalization ${log_normalization} --dataset ${dataset} --mode sample --model ${model} --langevin_step ${langevin_step} --samplingsteps ${samplingsteps} --n_samples 1 --per_step 250  --sigma_time ${sigma_time} --sigma_loc ${sigma_loc} --seq_len ${seq_len} --catalog_path ${catalog_path} --Mcut ${Mcut} --auxiliary_start ${auxiliary_start} --train_nll_start ${train_nll_start} --val_nll_start ${val_nll_start} --test_nll_start ${test_nll_start} --test_nll_end ${test_nll_end} --marked_output ${marked_output} --weight_path ${save_path}model_best.pkl


# python app.py --dataset ${dataset} --mode train --save_path ${save_path} --timesteps ${timesteps} --samplingsteps ${samplingsteps} --batch_size ${batch_size} --cuda_id ${cuda_id} --total_epochs ${total_epochs} --seq_len ${seq_len} --catalog_path ${catalog_path} --Mcut ${Mcut} --auxiliary_start ${auxiliary_start} --train_nll_start ${train_nll_start} --val_nll_start ${val_nll_start} --test_nll_start ${test_nll_start} --test_nll_end ${test_nll_end} --marked_output ${marked_output}
python app.py --day_number $1 --batch_size $2 --save_path ${save_path} --dataset ${dataset} --mode sample --model ${model} --timesteps ${timesteps}  --samplingsteps ${samplingsteps}  --seq_len ${seq_len} --catalog_path ${catalog_path} --Mcut ${Mcut} --auxiliary_start ${auxiliary_start} --train_nll_start ${train_nll_start} --val_nll_start ${val_nll_start} --test_nll_start ${test_nll_start} --test_nll_end ${test_nll_end} --marked_output ${marked_output} --weight_path ${save_path}model_best.pkl
