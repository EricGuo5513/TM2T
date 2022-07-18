from torch.utils.data import DataLoader
from utils.get_opt import get_opt
from text_loaders.m2t_nmt_model_dataset import M2TNMTGeneratedDataset
from utils.word_vectorizer import WordVectorizerV2
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def get_text_loader(opt_path, batch_size, ground_truth_dataset, device):
    opt = get_opt(opt_path, device)
    # print(opt.use_att)

    # Currently the configurations of two datasets are almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit':
        w_vectorizer = WordVectorizerV2('./glove', 'our_vab')
    else:
        raise KeyError('Dataset not recognized!!')
    print('Generating %s ...' % opt.name)

    if 'M2T' in opt.name:
        dataset = M2TNMTGeneratedDataset(opt, ground_truth_dataset, w_vectorizer)
    else:
        raise KeyError('Dataset not recognized!!')

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    print('Generated Dataset Loading Completed!!!')

    return motion_loader