import torch
import torch.nn as nn
import numpy as np
from DSTPP import GaussianDiffusion_ST, Transformer, Transformer_ST, Model_all, ST_Diffusion
from torch.optim import AdamW, Adam
import argparse
from scipy.stats import kstest
from DSTPP.Dataset import get_dataloader
import time
import setproctitle
from torch.utils.tensorboard import SummaryWriter
import datetime
import pickle
import os
from tqdm import tqdm
import random
import json
import sys
import math
import pandas as pd
import matplotlib.pyplot as plt

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def model_name():
    TIME = int(time.time())
    TIME = time.localtime(TIME)
    return time.strftime("%Y-%m-%d %H:%M:%S",TIME)

def normalization(x,MAX,MIN):
    return (x-MIN)/(MAX-MIN)

def denormalization(x,MAX,MIN,log_normalization=False):
    if log_normalization:
        return torch.exp(x.detach().cpu()*(MAX-MIN)+MIN)
    else:
        return x.detach().cpu()*(MAX-MIN)+MIN


def get_args():
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--seed', type=int, default=1234, help='')
    parser.add_argument('--model', type=str, default='DSTPP', help='')
    parser.add_argument('--seq_len', type=int, default = 100, help='')
    parser.add_argument('--mode', type=str, default='train', help='')
    parser.add_argument('--total_epochs', type=int, default=1000, help='')
    parser.add_argument('--machine', type=str, default='none', help='')
    parser.add_argument('--loss_type', type=str, default='l2',choices=['l1','l2','Euclid'], help='')
    parser.add_argument('--beta_schedule', type=str, default='cosine',choices=['linear','cosine'], help='')
    parser.add_argument('--dim', type=int, default=2, help='', choices = [1,2,3])
    parser.add_argument('--dataset', type=str, default='Earthquake',choices=['Earthquake','crime','football','ComCat','WHITE','SCEDC','SaltonSea','SanJac'], help='')
    parser.add_argument('--batch_size', type=int, default=64,help='')
    parser.add_argument('--timesteps', type=int, default=100, help='')
    parser.add_argument('--samplingsteps', type=int, default=100, help='')
    parser.add_argument('--objective', type=str, default='pred_noise', help='')
    parser.add_argument('--cuda_id', type=str, default='0', help='')
    parser.add_argument('--weight_path', type=str, default='./ModelSave/dataset_Earthquake_model_SMSTPP/model_300.pkl', help='')
    parser.add_argument('--save_path', type=str, help='')
    parser.add_argument('--Mcut', type=float, help='')  
    parser.add_argument('--catalog_path', type=str, help='')
    parser.add_argument('--auxiliary_start', type=lambda s : pd.to_datetime(s,format='%Y-%m-%d:%H:%M:%S'), help='')
    parser.add_argument('--train_nll_start', type=lambda s : pd.to_datetime(s,format='%Y-%m-%d:%H:%M:%S'), help='')
    parser.add_argument('--val_nll_start', type=lambda s : pd.to_datetime(s,format='%Y-%m-%d:%H:%M:%S'), help='')
    parser.add_argument('--test_nll_start', type=lambda s : pd.to_datetime(s,format='%Y-%m-%d:%H:%M:%S'), help='')
    parser.add_argument('--test_nll_end', type=lambda s : pd.to_datetime(s,format='%Y-%m-%d:%H:%M:%S'), help='')
    parser.add_argument('--marked_output', type=int, default=1, help='')
    parser.add_argument('--num_catalogs', type=int, default=10000, help='')
    parser.add_argument('--day_number', type=int, default=0, help='')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print(args)
    return args

opt = get_args()
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device(f"cuda:{opt.cuda_id}")

if opt.dataset == 'HawkesGMM':
    opt.dim=1

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.cuda_id)

# Force CPU usage if CUDA is not available
if not (torch.cuda.is_available()) or (opt.mode == 'sample'):
    print("CUDA not available, using CPU instead")
    device = torch.device("cpu")
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

def process_split(opt,df,pkl_filename):

    data_array = df.to_numpy()
    num_batches = len(data_array) // opt.seq_len
    batches = np.array_split(data_array[:num_batches * opt.seq_len], num_batches)

    batches_list = []
    for batch in batches:
        # Subtract the start time from all times in the batch
        start_time = batch[0, 0]-1  # The first time in the batch
        batch[:, 0] -= start_time  # Subtract start_time from all time values in the batch
        
        # Convert the batch to a list of lists
        batches_list.append(batch.tolist())

    leftover_rows = data_array[num_batches * opt.seq_len:]
    if len(leftover_rows) > 0:
        start_time = leftover_rows[0, 0]-1
        leftover_rows[:, 0] -= start_time
        batches_list.append(leftover_rows.tolist())


    with open(pkl_filename, 'wb') as file:
        pickle.dump(batches_list, file)

def preprocess_catalog(opt):

    df = pd.read_csv(
                    opt.catalog_path,
                    parse_dates=["time"],
                    dtype={"url": str, "alert": str},
                )
    df = df.sort_values(by='time')

    ### filter events by magnitude threshold

    df = df[df['magnitude']>=opt.Mcut]


    df = df[['time','x','y']]

    ### create train/val/test dfs
    aux_df = df[df['time']>=opt.auxiliary_start]
    aux_df = df[df['time']<opt.train_nll_start]

    # train_df = df[df['time']>=opt.train_nll_start]
    train_df = df[df['time']>=opt.auxiliary_start]
    train_df = train_df[train_df['time']< opt.val_nll_start]

    val_df = df[df['time']>=opt.val_nll_start]
    val_df = val_df[val_df['time']< opt.test_nll_start]

    test_df = df[df['time']>=opt.test_nll_start]
    test_df = test_df[test_df['time']< opt.test_nll_end]


    ## convert datetime to days

    train_df['time'] = (train_df['time']-train_df['time'].min()).dt.total_seconds() / (60*60*24)
    val_df['time'] = (val_df['time']-val_df['time'].min()).dt.total_seconds() / (60*60*24)
    test_df['time'] = (test_df['time']-test_df['time'].min()).dt.total_seconds() / (60*60*24)

    # List of DataFrames
    dfs = [train_df, val_df, test_df]

    # Process each DataFrame
    for i, df in enumerate(dfs):
        time_diffs = np.ediff1d(df['time'])

        # Identify the indices where the differences are less than or equal to 0
        indices_to_drop = np.where(time_diffs <= 0)[0] + 1

        indices_to_drop = df.index[indices_to_drop]

        # Drop the rows with the identified indices
        dfs[i] = df.drop(index=indices_to_drop)

    # Assign the processed DataFrames back
    train_df, val_df, test_df = dfs

    assert (np.ediff1d(train_df['time']) > 0).all()
    assert (np.ediff1d(val_df['time']) > 0).all()
    assert (np.ediff1d(test_df['time']) > 0).all()

    process_split(opt,train_df,'dataset/{}/seq_len_{}_data_train.pkl'.format(opt.dataset,opt.seq_len))
    process_split(opt,val_df,'dataset/{}/seq_len_{}_data_val.pkl'.format(opt.dataset,opt.seq_len))
    process_split(opt,test_df,'dataset/{}/seq_len_{}_data_test.pkl'.format(opt.dataset,opt.seq_len))

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

def azimuthal_equidistant_inverse(x, y, lat0, lon0, R=6371):
    """
    Inverse azimuthal equidistant projection.
    Converts (x, y) back to (lat, lon) from a center (lat0, lon0).
    
    Parameters:
    - x, y: Projected coordinates (km)
    - lat0, lon0: Center of the projection (degrees)
    - R: Radius of the sphere (default: Earth radius in km)
    
    Returns:
    - lat, lon: Geographic coordinates (degrees)
    """
    lat0, lon0 = deg2rad(lat0), deg2rad(lon0)
    
    r = np.sqrt(x**2 + y**2)
    c = r / R
    
    lat = np.where(r == 0, lat0, np.arcsin(np.cos(c) * np.sin(lat0) + (x * np.sin(c) * np.cos(lat0) / r)))
    lon = lon0 + np.arctan2(y * np.sin(c), r * np.cos(lat0) * np.cos(c) - x * np.sin(lat0) * np.sin(c))
    
    return rad2deg(lat), rad2deg(lon)


def create_test_day_dataloader(opt, day_number=opt.day_number, Max=None, Min=None, batch_size=32):

    df = pd.read_csv(
                    opt.catalog_path,
                    parse_dates=["time"],
                    dtype={"url": str, "alert": str},
                )
    df = df.sort_values(by='time')
    df = df[df['magnitude'] >= opt.Mcut]
    center_latitude = df['latitude'].mean()
    center_longitude = df['longitude'].mean()
    df = df[['time','x','y']]


    test_day_begin = opt.test_nll_start + pd.Timedelta(days=day_number)
    print('test_day_begin', test_day_begin)
    test_day_end = opt.test_nll_start + pd.Timedelta(days=day_number+1)
    print('test_day_end', test_day_end)
    test_day_df = df[df['time'] < test_day_begin]

    # Convert Timestamps to numeric days
    test_day_df['time'] = (test_day_df['time'] - test_day_df['time'].min()).dt.total_seconds() / (60*60*24)
    test_day_begin = (test_day_begin - df['time'].min()).total_seconds() / (60*60*24)
    test_day_end = (test_day_end - df['time'].min()).total_seconds() / (60*60*24)

    test_day_array = test_day_df.to_numpy()


    # Keep only the last seq_len rows if needed
    if len(test_day_array) > opt.seq_len:
        test_day_array = test_day_array[-opt.seq_len:]

    # Convert Timestamps to numeric days
    start_time = test_day_array[0, 0]  

    start_time_datetime = df['time'].min() + pd.Timedelta(days=start_time-1)

    print('start_time_datetime', start_time_datetime)

    test_day_array[:, 0] = ((test_day_array[:, 0] - start_time) + 1.0)
    start_time_float = (test_day_begin - start_time) + 1.0
    end_time_float = (test_day_end - start_time) + 1.0
    print('end_time_float', end_time_float)

    print('end_datetime', start_time_datetime + pd.Timedelta(days=end_time_float))

    # convert to list of lists
    test_day_array = [test_day_array.tolist()]*opt.num_catalogs

    test_day_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0]]+ i[1:] for index, i in enumerate(u)] for u in test_day_array]

    test_day_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_day_data]

    test_dayloader = get_dataloader(test_day_data, batch_size=batch_size, D=opt.dim, shuffle=False)
    print('Max & Min', (Max, Min))
    return test_dayloader, start_time_datetime, start_time_float, end_time_float, center_latitude, center_longitude


def data_loader(opt):

    f = open('dataset/{}/seq_len_{}_data_train.pkl'.format(opt.dataset,opt.seq_len),'rb')
    train_data = pickle.load(f)
    train_data = [[list(i) for i in u] for u in train_data]
    train_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0]]+ i[1:] for index, i in enumerate(u)] for u in train_data]

    f = open('dataset/{}/seq_len_{}_data_val.pkl'.format(opt.dataset,opt.seq_len),'rb')
    val_data = pickle.load(f)
    val_data = [[list(i) for i in u] for u in val_data]
    val_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0]]+ i[1:] for index, i in enumerate(u)] for u in val_data]

    f = open('dataset/{}/seq_len_{}_data_test.pkl'.format(opt.dataset,opt.seq_len),'rb')
    test_data = pickle.load(f)
    test_data = [[list(i) for i in u] for u in test_data]
    test_data = [[[i[0], i[0]-u[index-1][0] if index>0 else i[0]]+ i[1:] for index, i in enumerate(u)] for u in test_data]

    data_all = train_data+test_data+val_data

    Max, Min = [], []
    for m in range(opt.dim+2):
        if m > 0:
            Max.append(max([i[m] for u in data_all for i in u]))
            Min.append(min([i[m] for u in data_all for i in u]))
        else:
            Max.append(1)
            Min.append(0)

    assert Min[1] > 0
    
    train_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in train_data]
    test_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in test_data]
    val_data = [[[normalization(i[j], Max[j], Min[j]) for j in range(len(i))] for i in u] for u in val_data]

    trainloader = get_dataloader(train_data, opt.batch_size, D = opt.dim, shuffle=True)
    testloader = get_dataloader(test_data, len(test_data) if len(test_data)<=1000 else 1000, D = opt.dim, shuffle=False)
    valloader = get_dataloader(test_data, len(val_data) if len(val_data)<=1000 else 1000, D = opt.dim, shuffle=False)

    return trainloader, testloader, valloader, (Max,Min)


def Batch2toModel(batch, transformer):

    if opt.dim ==1:
        event_time_origin, event_time, lng = map(lambda x: x.to(device), batch)
        event_loc = lng.unsqueeze(dim=2)

    if opt.dim==2:
        event_time_origin, event_time, lng, lat = map(lambda x: x.to(device), batch)

        event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2)),dim=-1)

    if opt.dim==3:
        event_time_origin, event_time, lng, lat, height = map(lambda x: x.to(device), batch)

        event_loc = torch.cat((lng.unsqueeze(dim=2),lat.unsqueeze(dim=2), height.unsqueeze(dim=2)),dim=-1)

    event_time = event_time.to(device)
    event_time_origin = event_time_origin.to(device)
    event_loc = event_loc.to(device)
    
    enc_out, mask = transformer(event_loc, event_time_origin)

    enc_out_non_mask  = []
    event_time_non_mask = []
    event_loc_non_mask = []
    for index in range(mask.shape[0]):
        length = int(sum(mask[index]).item())
        if length>1:
            enc_out_non_mask += [i.unsqueeze(dim=0) for i in enc_out[index][:length-1]]
            event_time_non_mask += [i.unsqueeze(dim=0) for i in event_time[index][1:length]]
            event_loc_non_mask += [i.unsqueeze(dim=0) for i in event_loc[index][1:length]]

    enc_out_non_mask = torch.cat(enc_out_non_mask,dim=0)
    event_time_non_mask = torch.cat(event_time_non_mask,dim=0)
    event_loc_non_mask = torch.cat(event_loc_non_mask,dim=0)

    event_time_non_mask = event_time_non_mask.reshape(-1,1,1)
    event_loc_non_mask = event_loc_non_mask.reshape(-1,1,opt.dim)
    
    enc_out_non_mask = enc_out_non_mask.reshape(event_time_non_mask.shape[0],1,-1)

    return event_time_non_mask, event_loc_non_mask, enc_out_non_mask


def LR_warmup(lr, epoch_num, epoch_current):
    return lr * (epoch_current+1) / epoch_num


if __name__ == "__main__":
    
    setup_init(opt)
    setproctitle.setproctitle("Model-Training")

    print('dataset:{}'.format(opt.dataset))

    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.datetime.now())
    
    # Specify a directory for logging data 
    logdir = "./logs/{}_timesteps_{}".format( opt.dataset,  opt.timesteps)
    model_path = opt.save_path

    if not os.path.exists('./ModelSave'):
        os.mkdir('./ModelSave')

    if 'train' in opt.mode and not os.path.exists(model_path):
        os.mkdir(model_path)

    writer = SummaryWriter(log_dir = logdir,flush_secs=5)

    model= ST_Diffusion(
        n_steps=opt.timesteps,
        dim=1+opt.dim,
        condition = True,
        cond_dim=64
    ).to(device)


    diffusion = GaussianDiffusion_ST(
        model,
        loss_type = opt.loss_type,
        seq_length = 1+opt.dim,
        timesteps = opt.timesteps,
        sampling_timesteps = opt.samplingsteps,
        objective = opt.objective,
        beta_schedule = opt.beta_schedule
    ).to(device)

    transformer = Transformer_ST(
        d_model=64,
        d_rnn=256,
        d_inner=128,
        n_layers=4,
        n_head=4,
        d_k=16,
        d_v=16,
        dropout=0.1,
        device=device,
        loc_dim = opt.dim,
        CosSin = True
    ).to(device)

    Model = Model_all(transformer,diffusion)

    preprocess_catalog(opt)

    trainloader, testloader, valloader, (MAX,MIN) = data_loader(opt)

    if opt.mode == 'test' or opt.mode == 'sample':
        Model.load_state_dict(torch.load(opt.weight_path, map_location=device))
        print('Weight loaded!!')
    total_params = sum(p.numel() for p in Model.parameters())
    print(f"Number of parameters: {total_params}")

    warmup_steps = 5
    
    # training
    optimizer = AdamW(Model.parameters(), lr = 1e-3, betas = (0.9, 0.99))
    step, early_stop = 0, 0
    min_loss_test = 1e20
    for itr in range(opt.total_epochs):

        print('epoch:{}'.format(itr))

        if (itr % 10==0) or (opt.mode == 'test') or (opt.mode == 'sample'):
            print('Evaluate!')
            with torch.no_grad():
                
                Model.eval()

                if opt.mode == 'sample':
                    test_day_loader, start_time_datetime, start_time_float, end_time_float, center_lat, center_lon = create_test_day_dataloader(opt, day_number=opt.day_number, Max=MAX, Min=MIN,batch_size=opt.batch_size)

                    print('Sampling!')
                    for idx, batch in enumerate(test_day_loader):
                        print('Batch {} of {}'.format(idx, len(test_day_loader)))
                        which_under_end_time = torch.ones(1, batch[0].shape[0], dtype=torch.bool)
                        # create a list of indexes that are alive based on the index within the whole test data
                        indexes_alive = list(range(idx*opt.batch_size, idx*opt.batch_size+ batch[0].shape[0]))

                        # create an empty df to store the generated events
                        gen_df = pd.DataFrame(columns=['mag','time_string','x','y','depth','catalog_id'])
                        
                        round_number = 0
                        # while at least one random sequence is whithin the forecast horizon
                        while which_under_end_time.sum() > 0:

                            round_number += 1
                            print('Generation:', round_number)
                            sampled_record_all = []

                            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

                            # isolate last event in the batch
                            enc_out_non_mask = enc_out_non_mask[opt.seq_len-2::opt.seq_len-1,:,:]


                            sampled_seq = Model.diffusion.sample(batch_size = enc_out_non_mask.shape[0],cond=enc_out_non_mask)
                            sampled_seq_temporal_all=(sampled_seq[:,0,:1].detach().cpu() + MIN[1]) * (MAX[1]-MIN[1])

                            locs = (denormalization(sampled_seq[:,:,-2:], torch.tensor([MAX[-2:]]), torch.tensor([MIN[-2:]])))
                            # print('Max & Min', (MAX, MIN))
                            # print(locs.shape)
                            # plt.scatter(locs[:,:,0], locs[:,:,1])
                            # plt.show()

                            # sampled_seq_spatial_all=((sampled_seq[:,0,-opt.dim:].detach().cpu() + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))).unsqueeze(dim=1)
                            sampled_seq_spatial_all = denormalization(sampled_seq[:,:,-2:], torch.tensor([MAX[-2:]]), torch.tensor([MIN[-2:]]))

                            # convert the generated events to datetime
                            last_event_times = batch[0][:,-1]
                            gen_event_times = last_event_times + sampled_seq_temporal_all.cpu().t()
                            gen_event_datetimes = start_time_datetime + pd.to_timedelta(gen_event_times.numpy().flatten(),unit='D')
                            print(gen_event_datetimes)

                            gen_events = torch.cat((gen_event_times.t().unsqueeze(dim=2), sampled_seq.cpu()), dim=-1)

                            which_over_start_time = (gen_event_times > start_time_float)[0]
                            which_under_end_time = (gen_event_times < end_time_float)[0]
                            # only keep indexes which unders the end time return empty list if none
                            indexes_alive = [i for idx, i in enumerate(indexes_alive) if which_under_end_time[idx].item()]

                            # Append the generated events to the DataFrame using pd.concat
                            for idx, i in enumerate(indexes_alive):
                                if (which_under_end_time[idx].item()) and (which_over_start_time[idx].item()):
                                    new_row = pd.DataFrame([{
                                        'mag': opt.Mcut,
                                        'time_string': gen_event_datetimes[idx].strftime('%Y-%m-%dT%H:%M:%S'),
                                        'x': sampled_seq_spatial_all[idx, 0, 0].item(),
                                        'y': sampled_seq_spatial_all[idx, 0, 1].item(),
                                        'depth': 0,
                                        'catalog_id': i
                                    }])
                                    gen_df = pd.concat([gen_df, new_row], ignore_index=True) 
                            

                            # modify the batch to include the generated events
                            batch_list = list(batch)
                            # Add to the end of the batch
                            for i in range(len(batch_list)):
                                batch_list[i] = torch.cat((batch_list[i], gen_events[:, :, i]), dim=1)
                                # remove the first element of the batch
                                batch_list[i] = batch_list[i][:, 1:]
                                # remove which over the end time
                                batch_list[i] = batch_list[i][which_under_end_time,:]

                            # Convert the list back to a tuple 
                            batch = tuple(batch_list)   


                        # perform azimuthal equidistant projection inverse on x and y  
                        gen_df['lat'], gen_df['lon'] = azimuthal_equidistant_inverse(gen_df['x'], gen_df['y'], center_lat, center_lon)

                        # only keep lat lon mag','time_string',,'depth','catalog_id
                        gen_df = gen_df[['lon','lat','mag','time_string','depth','catalog_id']]

                        # sort the df by catalog_id then time_string
                        gen_df = gen_df.sort_values(by=['catalog_id','time_string'])

                        # path_to_forecasts = './'
                        path_to_forecasts = '/user/work/ss15859/'

                        # write batch to csv
                        if not os.path.exists(path_to_forecasts +'DSTPP_daily_forecasts'):
                            os.mkdir(path_to_forecasts +'DSTPP_daily_forecasts')

                        if not os.path.exists(path_to_forecasts +'DSTPP_daily_forecasts/{}'.format(opt.dataset)):
                            os.mkdir(path_to_forecasts +'DSTPP_daily_forecasts/{}'.format(opt.dataset))
                        
                        if not os.path.exists(path_to_forecasts +'DSTPP_daily_forecasts/{}/CSEP_day_{}.csv'.format(opt.dataset, opt.day_number)):
                            gen_df.to_csv(path_to_forecasts +'DSTPP_daily_forecasts/{}/CSEP_day_{}.csv'.format(opt.dataset, opt.day_number), index=False)
                        else:
                            gen_df.to_csv(path_to_forecasts +'DSTPP_daily_forecasts/{}/CSEP_day_{}.csv'.format(opt.dataset, opt.day_number), mode='a', header=False, index=False)
                                                                                                    

                    ### sampling code here

                if opt.mode == 'sample':
                    print('Sampling Done!')
                    break    
                
                # validation set
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for batch in valloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

                    sampled_seq = Model.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)

                    # sampled_seq_temporal_all, sampled_seq_spatial_all = [], []
                    # for _ in range(100):
                    #     sampled_seq = Model.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)
                    #     sampled_seq_temporal_all.append((sampled_seq[:,0,:1].detach().cpu() + MIN[1]) * (MAX[1]-MIN[1]))
                    #     sampled_seq_spatial_all.append(((sampled_seq[:,0,-opt.dim:].detach().cpu() + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))).unsqueeze(dim=1))

                    loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)

                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
                    
                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial

                    loss_test_all += loss.item() * event_time_non_mask.shape[0]
                    
                    # real = (event_time_non_mask[:,0,:].detach().cpu() + MIN[1]) * (MAX[1]-MIN[1])
                    # gen = (sampled_seq[:,0,:1].detach().cpu() + MIN[1]) * (MAX[1]-MIN[1])
                    # assert real.shape==gen.shape
                    # mae_temporal += torch.abs(real-gen).sum().item()
                    # rmse_temporal += ((real-gen)**2).sum().item()
                    
                    # real = event_loc_non_mask[:,0,:].detach().cpu()
                    # assert real.shape[1:] == torch.tensor(MIN[2:]).shape
                    # real = (real + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))
                    # gen = sampled_seq[:,0,-opt.dim:].detach().cpu()
                    # gen = (gen + torch.tensor([MIN[2:]])) * (torch.tensor([MAX[2:]])-torch.tensor([MIN[2:]]))
                    # assert real.shape==gen.shape
                    # mae_spatial += torch.sqrt(torch.sum((real-gen)**2,dim=-1)).sum().item()


                    total_num += event_time_non_mask.shape[0]

                    # assert gen.shape[0] == event_time_non_mask.shape[0]

                if loss_test_all > min_loss_test:
                    early_stop += 1
                    if early_stop >= 100:
                        print('Early Stop!!')
                        break
                
                else:
                    print('Model Updated!!')
                    torch.save(Model.state_dict(), opt.save_path + 'model_best.pkl')
                    early_stop = 0
                
                torch.save(Model.state_dict(), model_path+'model_{}.pkl'.format(itr))

                min_loss_test = min(min_loss_test, loss_test_all)

                writer.add_scalar(tag='Evaluation/loss_val',scalar_value=loss_test_all/total_num,global_step=itr)

                writer.add_scalar(tag='Evaluation/NLL_val',scalar_value=vb_test_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_temporal_val',scalar_value=vb_test_temporal_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_spatial_val',scalar_value=vb_test_spatial_all/total_num,global_step=itr)

                writer.add_scalar(tag='Evaluation/mae_temporal_val',scalar_value=mae_temporal/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/rmse_temporal_val',scalar_value=np.sqrt(rmse_temporal/total_num),global_step=itr)
                # writer.add_scalar(tag='Evaluation/rmse_temporal_mean_val',scalar_value=np.sqrt(rmse_temporal_mean/total_num),global_step=itr)
                
                writer.add_scalar(tag='Evaluation/distance_spatial_val',scalar_value=mae_spatial/total_num,global_step=itr)
                # writer.add_scalar(tag='Evaluation/distance_spatial_mean_val',scalar_value=mae_spatial_mean/total_num,global_step=itr)

                # test set
                loss_test_all, vb_test_all, vb_test_temporal_all, vb_test_spatial_all = 0.0, 0.0, 0.0, 0.0
                mae_temporal, rmse_temporal, mae_spatial, mae_lng, mae_lat, total_num = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for batch in testloader:
                    event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)

                    sampled_seq = Model.diffusion.sample(batch_size = event_time_non_mask.shape[0],cond=enc_out_non_mask)

                    loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)

                    vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)
                    
                    vb_test_all += vb
                    vb_test_temporal_all += vb_temporal
                    vb_test_spatial_all += vb_spatial

                    loss_test_all += loss.item() * event_time_non_mask.shape[0]

                    total_num += event_time_non_mask.shape[0]

                writer.add_scalar(tag='Evaluation/loss_test',scalar_value=loss_test_all/total_num,global_step=itr)

                writer.add_scalar(tag='Evaluation/NLL_test',scalar_value=vb_test_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_temporal_test',scalar_value=vb_test_temporal_all/total_num,global_step=itr)
                writer.add_scalar(tag='Evaluation/NLL_spatial_test',scalar_value=vb_test_spatial_all/total_num,global_step=itr)
                
        if itr < warmup_steps:
            for param_group in optimizer.param_groups:
                lr = LR_warmup(1e-3, warmup_steps, itr)
                param_group["lr"] = lr

        else:
            for param_group in optimizer.param_groups:
                lr = 1e-3- (1e-3 - 5e-5)*(itr-warmup_steps)/opt.total_epochs
                param_group["lr"] = lr

        writer.add_scalar(tag='Statistics/lr',scalar_value=lr,global_step=itr)

        Model.train()

        loss_all, vb_all, vb_temporal_all, vb_spatial_all, total_num = 0.0, 0.0, 0.0, 0.0, 0.0
        for batch in trainloader:

            event_time_non_mask, event_loc_non_mask, enc_out_non_mask = Batch2toModel(batch, Model.transformer)
            loss = Model.diffusion(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1),enc_out_non_mask)

            optimizer.zero_grad()
            loss.backward()

            loss_all += loss.item() * event_time_non_mask.shape[0]
            vb, vb_temporal, vb_spatial = Model.diffusion.NLL_cal(torch.cat((event_time_non_mask,event_loc_non_mask),dim=-1), enc_out_non_mask)

            vb_all += vb
            vb_temporal_all += vb_temporal
            vb_spatial_all += vb_spatial

            writer.add_scalar(tag='Training/loss_step',scalar_value=loss.item(),global_step=step)

            torch.nn.utils.clip_grad_norm_(Model.parameters(), 1.)
            optimizer.step() 
            
            step += 1

            total_num += event_time_non_mask.shape[0]

        with torch.cuda.device("cuda:{}".format(opt.cuda_id)):
            torch.cuda.empty_cache()

        writer.add_scalar(tag='Training/loss_epoch',scalar_value=loss_all/total_num,global_step=itr)

        writer.add_scalar(tag='Training/NLL_epoch',scalar_value=vb_all/total_num,global_step=itr)
        writer.add_scalar(tag='Training/NLL_temporal_epoch',scalar_value=vb_temporal_all/total_num,global_step=itr)
        writer.add_scalar(tag='Training/NLL_spatial_epoch',scalar_value=vb_spatial_all/total_num,global_step=itr)
