from finetuning import datasets
from finetuning.utils import train_test_split
from transplant.utils import save_pkl
data = datasets.get_ptb_xl_data(
   db_dir='data/ptbxl',
   fs=250,  # keep sampling frequency the same as Icentia11k
   pad=16384,  # zero-pad recordings to keep the same length at about 65 seconds
   normalize=True)  # normalize each recording with mean and std computed over the entire dataset
# maintain class ratio across both train and test sets by using the `stratify` argument
train_set, test_set = train_test_split(
   data, test_size=0.2)
save_pkl('data/ptbxl_train.pkl', **train_set)
save_pkl('data/ptbxl_test.pkl', **test_set)

