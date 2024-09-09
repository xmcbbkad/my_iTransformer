import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import json

warnings.filterwarnings('ignore')

def in_date_list(file, date_list):
    for item in date_list:
        if file.find(item) >=0:
            return True
    return False

class Dataset_Stock_Price(Dataset):
    def __init__(self, root_path, data_config_file, flag, size, features="MS"):
        self.seq_len = size[0]
        self.pred_len = size[2]
        
        self.df_list = []
        self.df_index = []
        #import pdb; pdb.set_trace()
        csv_files = []

        #dict_data_config = json.load(open(os.path.join(root_path, data_config_file), 'r'))
        dict_data_config = json.load(open(data_config_file, 'r'))
        list_date = dict_data_config[flag]        

        for root, dirs,files in os.walk(root_path):
            for file in files:
                if file.lower().endswith(".csv") and in_date_list(file, list_date):
                    csv_files.append(os.path.join(root, file))
        
        csv_files.sort()
        for f_index in range(len(csv_files)):
            print("{}/{}".format(f_index, len(csv_files)))
            df_this = pd.read_csv(csv_files[f_index]).iloc[::-1].reset_index(drop=True)
            df_this = df_this[['date', 'open', 'high', 'low', 'close']]
            #data = df_this.values
            #self.df_list.append(data)
            self.df_list.append(df_this)
            if len(df_this) < self.seq_len + self.pred_len:
                continue
            #for i in range(len(data)-self.pred_len):
            for i in range(len(df_this)-self.seq_len-self.pred_len):
                self.df_index.append([len(self.df_list)-1, i])
        #self.data_x = data[0: len(df_raw)-self.seq_len-self.pred_len]
        #self.data_y = data[0: len(df_raw)-self.seq_len-self.pred_len]
    
    def __getitem__(self, index):
        #import pdb; pdb.set_trace()
        index_1 = self.df_index[index][0]
        data = self.df_list[index_1]
        
        index_2 = self.df_index[index][1]

        s_begin = index_2
        s_end = s_begin + self.seq_len

        r_begin = s_end
        r_end = r_begin + self.pred_len

        seq_x = data[s_begin:s_end]

        new_columns = ["norm_open", "norm_high", "norm_low", "norm_close"]
       
        base_index = seq_x.index[0]

        seq_x[new_columns] = 1.0
        for i in range(seq_x.shape[0]):
            row_index = seq_x.index[i]
            seq_x.loc[row_index, "norm_open"] = (seq_x.loc[row_index, "open"]/seq_x.loc[base_index, "open"]-1)*100
            seq_x.loc[row_index, "norm_high"] = (seq_x.loc[row_index, "high"]/seq_x.loc[base_index, "open"]-1)*100
            seq_x.loc[row_index, "norm_low"] = (seq_x.loc[row_index, "low"]/seq_x.loc[base_index, "open"]-1)*100
            seq_x.loc[row_index, "norm_close"] = (seq_x.loc[row_index, "close"]/seq_x.loc[base_index, "open"]-1)*100
            
        
        seq_y = data[r_begin:r_end]
        seq_y[new_columns] = 1.0
        for i in range(seq_y.shape[0]):
            row_index = seq_y.index[i]
            seq_y.loc[row_index, "norm_open"] = (seq_y.loc[row_index, "open"]/seq_x.loc[base_index, "open"]-1)*100
            seq_y.loc[row_index, "norm_high"] = (seq_y.loc[row_index, "high"]/seq_x.loc[base_index, "open"]-1)*100
            seq_y.loc[row_index, "norm_low"] = (seq_y.loc[row_index, "low"]/seq_x.loc[base_index, "open"]-1)*100
            seq_y.loc[row_index, "norm_close"] = (seq_y.loc[row_index, "close"]/seq_x.loc[base_index, "open"]-1)*100
        
        seq_x = seq_x[new_columns].values
        seq_y = seq_y[new_columns].values 
        return seq_x, seq_y

    def __len__(self):
        return len(self.df_index)



class Dataset_Stock_UpOrDown_realtime(Dataset):
    def __init__(self, root_path, data_config_file, flag, size, features="MS"):
        self.seq_len = size[0]
        self.pred_len = size[2]
        
        self.df_list = []
        self.df_index = []
        #import pdb; pdb.set_trace()
        csv_files = []

        #dict_data_config = json.load(open(os.path.join(root_path, data_config_file), 'r'))
        dict_data_config = json.load(open(data_config_file, 'r'))
        list_date = dict_data_config[flag]        

        for root, dirs,files in os.walk(root_path):
            for file in files:
                if file.lower().endswith(".csv") and in_date_list(file, list_date):
                    csv_files.append(os.path.join(root, file))
        
        csv_files.sort()
        for f_index in range(len(csv_files)):
            #import pdb; pdb.set_trace()
            #df_this = pd.read_csv(csv_files[f_index]).iloc[::-1].reset_index(drop=True)
            df_this = pd.read_csv(csv_files[f_index])
            df_this = df_this[['date', 'open', 'high', 'low', 'close']]
            df_this[["up_or_down"]] = -1
            df_this[["up_or_down_date"]] = ""
            
            for i in range(0, df_this.shape[0]):
                print("{}/{}  {}/{}".format(i, df_this.shape[0], f_index, len(csv_files)))
                for j in range(i+1, df_this.shape[0]):
                    if df_this.loc[j, "low"]/df_this.loc[i, "close"] <= 0.995:
                        df_this.loc[i, "up_or_down"] = 0
                        df_this.loc[i, "up_or_down_date"] = df_this.loc[j, "date"]
                        break
                    elif df_this.loc[j, "high"]/df_this.loc[i, "close"] >= 1.005:
                        df_this.loc[i, "up_or_down"] = 1
                        df_this.loc[i, "up_or_down_date"] = df_this.loc[j, "date"]
                        break
                    elif j == df_this.shape[0] - 1:                    
                        if df_this.loc[j, "low"] <= df_this.loc[i, "close"]:
                            df_this.loc[i, "up_or_down"] = 0
                            df_this.loc[i, "up_or_down_date"] = df_this.loc[j, "date"]
                        else:
                            df_this.loc[i, "up_or_down"] = 1
                            df_this.loc[i, "up_or_down_date"] = df_this.loc[j, "date"]

            self.df_list.append(df_this)
            if len(df_this) < self.seq_len + self.pred_len:
                continue
            for i in range(len(df_this)-self.seq_len-self.pred_len):
                self.df_index.append([len(self.df_list)-1, i])


    def __getitem__(self, index):
        import pdb; pdb.set_trace()
        index_1 = self.df_index[index][0]
        data = self.df_list[index_1]
        
        index_2 = self.df_index[index][1]

        s_begin = index_2
        s_end = s_begin + self.seq_len

        seq_x = data[s_begin:s_end]

        new_columns = ["norm_open", "norm_high", "norm_low", "norm_close"]
       
        base_index = seq_x.index[0]

        seq_x[new_columns] = 1.0
        for i in range(seq_x.shape[0]):
            row_index = seq_x.index[i]
            seq_x.loc[row_index, "norm_open"] = (seq_x.loc[row_index, "open"]/seq_x.loc[base_index, "open"]-1)*100
            seq_x.loc[row_index, "norm_high"] = (seq_x.loc[row_index, "high"]/seq_x.loc[base_index, "open"]-1)*100
            seq_x.loc[row_index, "norm_low"] = (seq_x.loc[row_index, "low"]/seq_x.loc[base_index, "open"]-1)*100
            seq_x.loc[row_index, "norm_close"] = (seq_x.loc[row_index, "close"]/seq_x.loc[base_index, "open"]-1)*100
            
        
        
        seq_y = seq_x.iloc[-1]["up_or_down"]
        seq_x = seq_x[new_columns].values
        return seq_x, seq_y

    def __len__(self):
        return len(self.df_index)


class Dataset_Stock_UpOrDown(Dataset):
    def __init__(self, root_path, data_config_file, flag, size, features="MS"):
        self.seq_len = size[0]
        self.pred_len = size[2]
        
        self.df_list = []
        self.df_index = []
        #import pdb; pdb.set_trace()
        csv_files = []

        #dict_data_config = json.load(open(os.path.join(root_path, data_config_file), 'r'))
        dict_data_config = json.load(open(data_config_file, 'r'))
        list_date = dict_data_config[flag]        

        for root, dirs,files in os.walk(root_path):
            for file in files:
                if file.lower().endswith(".csv") and in_date_list(file, list_date):
                    csv_files.append(os.path.join(root, file))
        
        csv_files.sort()
        for f_index in range(len(csv_files)):
            #import pdb; pdb.set_trace()
            #df_this = pd.read_csv(csv_files[f_index]).iloc[::-1].reset_index(drop=True)
            df_this = pd.read_csv(csv_files[f_index])
            df_this = df_this[['date', 'open', 'high', 'low', 'close', 'up_or_down']]
            
            self.df_list.append(df_this)
            if len(df_this) < self.seq_len + self.pred_len:
                continue
            for i in range(len(df_this)-self.seq_len-self.pred_len):
                self.df_index.append([len(self.df_list)-1, i])


    def __getitem__(self, index):
        #import pdb; pdb.set_trace()
        index_1 = self.df_index[index][0]
        data = self.df_list[index_1]
        
        index_2 = self.df_index[index][1]

        s_begin = index_2
        s_end = s_begin + self.seq_len

        seq_x = data[s_begin:s_end]

        new_columns = ["norm_open", "norm_high", "norm_low", "norm_close"]
       
        base_index = seq_x.index[0]

        seq_x[new_columns] = 1.0
        for i in range(seq_x.shape[0]):
            row_index = seq_x.index[i]
            seq_x.loc[row_index, "norm_open"] = (seq_x.loc[row_index, "open"]/seq_x.loc[base_index, "open"]-1)*100
            seq_x.loc[row_index, "norm_high"] = (seq_x.loc[row_index, "high"]/seq_x.loc[base_index, "open"]-1)*100
            seq_x.loc[row_index, "norm_low"] = (seq_x.loc[row_index, "low"]/seq_x.loc[base_index, "open"]-1)*100
            seq_x.loc[row_index, "norm_close"] = (seq_x.loc[row_index, "close"]/seq_x.loc[base_index, "open"]-1)*100
            
        
        
        seq_y = seq_x.iloc[-1]["up_or_down"]
        seq_x = seq_x[new_columns].values
        return seq_x, seq_y

    def __len__(self):
        return len(self.df_index)




def add_upordown_feature(input_file, output_file):
    df_this = pd.read_csv(input_file)
    df_this = df_this[['date', 'open', 'high', 'low', 'close']]
    df_this[["up_or_down"]] = -1
    df_this[["up_or_down_date"]] = ""
    
    for i in range(0, df_this.shape[0]):
        for j in range(i+1, df_this.shape[0]):
            #print(i, j)
            if df_this.loc[j, "low"]/df_this.loc[i, "close"] <= 0.995:
                df_this.loc[i, "up_or_down"] = 0
                df_this.loc[i, "up_or_down_date"] = df_this.loc[j, "date"]
                break
            elif df_this.loc[j, "high"]/df_this.loc[i, "close"] >= 1.005:
                df_this.loc[i, "up_or_down"] = 1
                df_this.loc[i, "up_or_down_date"] = df_this.loc[j, "date"]
                break
            elif j == df_this.shape[0] - 1:                    
                if df_this.loc[j, "low"] <= df_this.loc[i, "close"]:
                    df_this.loc[i, "up_or_down"] = 0
                    df_this.loc[i, "up_or_down_date"] = df_this.loc[j, "date"]
                else:
                    df_this.loc[i, "up_or_down"] = 1
                    df_this.loc[i, "up_or_down_date"] = df_this.loc[j, "date"]

   
    df_this.to_csv(output_file, index=False)


def add_upordown_feature_to_all(input_dir, output_dir):
    for root, dirs,files in os.walk(input_dir):
        for file in files:
            if not file.lower().endswith(".csv"):
                continue
            old_file_path = os.path.join(root, file)
            new_file_path = old_file_path.replace(input_dir, output_dir)
            print(old_file_path)
            print(new_file_path)
            os.makedirs(os.path.dirname(new_file_path), exist_ok=True)
            add_upordown_feature(old_file_path, new_file_path)

if __name__ == "__main__":
    #input_file = "/my_iTransformer/dataset/stock/taobao/AAPL.csv"
    #output_file = "/my_iTransformer/dataset/stock/taobao/AAPL__with_up_or_down.csv"
    #add_upordown_feature(input_file, output_file)
    
    
    input_dir = "/my_iTransformer/dataset/stock/taobao/"
    output_dir = "/my_iTransformer/dataset/stock/taobao_0.5_up_or_down/"
    add_upordown_feature_to_all(input_dir, output_dir)
