from easydict import EasyDict as edict

# init
__C_MULDATASET = edict()

cfg_data = __C_MULDATASET

__C_MULDATASET.STD_SIZE = (768,1024)
__C_MULDATASET.TRAIN_SIZE = 768
__C_MULDATASET.DATA_PATH = '../ProcessedData/MULDATASET'

#ImageNet
# __C_MULDATASET.MEAN_STD = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
#muldataset
__C_MULDATASET.MEAN_STD = ([0.4212788939476013, 0.3997511863708496, 0.39277446269989014],[0.24838325381278992, 0.2488340586423874, 0.24822652339935303])

__C_MULDATASET.LABEL_FACTOR = 1
__C_MULDATASET.LOG_PARA = 100.

__C_MULDATASET.RESUME_MODEL = ''#model path
__C_MULDATASET.TRAIN_BATCH_SIZE = 2 #imgs

__C_MULDATASET.VAL_BATCH_SIZE = 1 #


