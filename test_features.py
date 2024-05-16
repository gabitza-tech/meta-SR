from __future__ import print_function
import os
import pandas as pd
from sklearn.metrics import roc_curve
import sys 
import pickle as pkl
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import torch.nn.functional as F
from torch.autograd import Variable

from generator.SR_Dataset import *
from str2bool import str2bool
from generator.DB_wav_reader import read_feats_structure
from model.model import background_resnet

import os
import numpy as np
import feat_extract.constants as c
from python_speech_features import fbank,delta
import scipy.io.wavfile

def normalize_frames(m,Scale=False):
    if Scale:
        return (m - np.mean(m, axis = 0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))

def load_model(path, n_classes=5994):
    model = background_resnet(num_classes=n_classes)
    print('=> loading checkpoint')
    # load pre-trained parameters
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])

    return model

def extract_MFB(filename):
    sr, audio = scipy.io.wavfile.read(filename)
    features, energies = fbank(audio, samplerate=c.SAMPLE_RATE, nfilt=c.FILTER_BANK, winlen=0.025, winfunc=np.hamming)

    if c.USE_LOGSCALE:
        features = 20 * np.log10(np.maximum(features,1e-5))

    features = normalize_frames(features, Scale=False)

    return features

def get_d_vector(filename, model, use_cuda=True):
    input = extract_MFB(filename)
    TT = ToTensorTestInput()  # torch tensor:(1, n_dims, n_frames)
    input = TT(input)  # size : (n_frames, 1, n_filter, T)
    input = Variable(input)
    with torch.no_grad():
        if use_cuda:
            # load gpu
            input = input.cuda()
        activation = model(input)

    return activation

def compute_embeddings(model, wav_scp, outdir, use_cuda=True):
    """Compute speaker embeddings.

    Arguments
    ---------
    params: dict
        The parameter files storing info about model, data, etc
    wav_scp : str
        The wav.scp file in Kaldi, in the form of "$utt $wav_path"
    outdir: str
        The output directory where we store the embeddings in per-
        numpy manner.
    """
    model.eval()
    if use_cuda:
        model.to(torch.device("cuda:0"))

    with open(wav_scp, "r") as f:
        lines = f.readlines()

    out_dict = {}
    out_dict['concat_labels'] = []
    out_dict['concat_slices'] = []
    out_dict['concat_patchs'] = []
    out_dict['concat_features'] = []

    for i in tqdm(range(len(lines))):
            
        data = json.loads(lines[i].strip())
        
        if 'patch' in data.keys():
            out_dict['concat_patchs'].append(data['patch'])
        else:
            out_dict['concat_patchs'].append(data['file_id'])

        out_dict['concat_labels'].append(data['label'])
        out_dict['concat_slices'].append(data['file_id'])       
        
        with torch.no_grad():
            embeddings = get_d_vector(data['audio_filepath'], model, use_cuda)
            out_embedding = embeddings.detach().cpu().numpy()

        if not os.path.exists(outdir):
            os.mkdir(outdir)
        
        out_file = "{}/{}_metasr_embs.pkl".format(outdir, os.path.splitext(os.path.basename(wav_scp))[0])
        out_dict['concat_features'].append(np.asarray(out_embedding))
    
    out_dict['concat_features'] = np.concatenate(out_dict['concat_features'],axis=0)
    print(out_dict['concat_features'].shape)
    pkl.dump(out_dict, open(out_file, 'wb'))

if __name__ == "__main__":
    model = load_model('saved_model/checkpoint_100.pth')
    
    manifest = sys.argv[1]
    out_dir = sys.argv[2]

    compute_embeddings(model,manifest,out_dir)