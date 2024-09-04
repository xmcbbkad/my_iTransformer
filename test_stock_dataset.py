from data_provider.data_loader import Dataset_Custom
from data_provider.stock_data_loader import Dataset_Stock_Price, Dataset_Stock_UpOrDown
from torch.utils.data import DataLoader

#data_set = Dataset_Custom(root_path="/iTransformer/dataset/traffic", size=[96, 0, 96], features="MS", data_path="traffic.csv", timeenc=1)
#print(len(data_set))
#print(data_set[0])

data_set = Dataset_Stock_UpOrDown(root_path="dataset/stock/taobao/TSLA", data_config_file="dataset/stock/TSLA.json", flag="train", size=[30,0,30], features='MS')
print(len(data_set))
print(data_set[30])

#data_set = Dataset_Stock_Price(root_path="dataset/stock/taobao/TSLA", data_config_file="TSLA.json", flag="train", size=[30,0,30], features='MS')
#data_set = Dataset_Stock_Price(root_path="dataset/stock/taobao/TSLA_test", size=[30,0,30])
#data_set = Dataset_Stock_Price(root_path="dataset/stock/taobao/TSLA/", size=[30,0,30])
#print(len(data_set))
#print(data_set[340])
#print(data_set[10])
#print(data_set[11])

#for i in range(len(data_set)):
#    print(i, data_set[i][0].shape, data_set[i][1].shape)


#data_loader = DataLoader(
#    data_set,
#    batch_size=16,
#    shuffle=True,
#    num_workers=10,
#    drop_last=True
#)
#
#for i, (batch_x, batch_y) in enumerate(data_loader):
#    print(i)
