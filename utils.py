# ------------------------- I am a split line \(ow <) -------------------------
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim
import torch.nn.functional as func
import os

# ------------------------- I am a split line \(ow <) -------------------------
import time
import copy

def main():

    pass

# 将整个文件夹下所有的文件全部加载。
def load_all(dir):
    filenames=os.listdir(dir)
    filelist=[]
    for filename in filenames:
        if not os.path.isdir(filename):
            if os.path.splitext(filename)[-1]==".pth":
                filelist.append(os.path.join(dir,filename))
    print(filelist)
    return load_file_to_dataset(filelist)

# 从单/多个文件加载Dataset
def load_file_to_dataset(data_path: str or list):
    """将一个或者多个通过`process`程序生成的`.pth`文件作为本程序定义的DataSet加载到内存。

    Args:
        data_path (str or list): 数据文件路径

    Raises:
        ValueError: 输入类型不正确时raise一下error

    Returns:
        EEGDataSet: 本文件定义的数据集类型实例
    """
    if isinstance(data_path, str):
        dataset = EEGDataSet(path=data_path)
    elif isinstance(data_path, list):
        sets = []
        for path in data_path:
            sets.append(EEGDataSet(path))
        dataset = join_datasets(sets)
    else:
        raise ValueError("Argument not valid.")
    return dataset


# 将Dataset划分训练集，验证集和测试集
def split_dataset(dataset, split_rate=[0.7, 0.2, 0.1], rdn_seed=42):
    # 随机生成器
    generator = torch.Generator().manual_seed(rdn_seed)
    # 划分的子集
    subsets = random_split(dataset, split_rate, generator)
    return subsets


# 将一个list中的dataset全部合并
def join_datasets(datasets: list):
    # 通过创建一个新的DataSet对象
    new = EEGDataSet()
    new.info = datasets[0].info.copy()
    # 再把待合并的DataSet中的data合并到一起。
    for dataset in datasets:
        if dataset.info != new.info:
            print('Warning: Joining datasets with different attributes')
        new.data += copy.deepcopy(dataset.data)
    return new


# 自定义数据集类型
class EEGDataSet(torch.utils.data.Dataset):

    def __init__(self, path: str = None,to_cuda:bool=False) -> None:
        if path:
            self.loaded = torch.load(path)
            self.info = self.loaded['info']
            self.data = self.loaded['data']
        else:
            self.info = {}
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # (230414)形状要改一下，原本储存为[16*500]，这里要[500*16]，
        # 因为LSTM层输入要求第二层就是序列
        return self.data[idx]['data'].t(), self.data[idx]['label']
    
if __name__=="__main__":
    main()