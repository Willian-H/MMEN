import os
import argparse

from utils.functions import Storage

class ConfigRegression():
    def __init__(self, args):
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            # multi-task
            'v1': self.__V1,
            'v2': self.__V1,
            'v3': self.__V3,
            'v4': self.__V3,
            'v5': self.__V5,
            'v6': self.__V6,
            'v7': self.__V7,
            'v8': self.__V8,
        }
        # hyper parameters for datasets
        HYPER_DATASET_MAP = self.__datasetCommonParams()

        # normalize
        model_name = str.lower(args.modelName)
        dataset_name = str.lower(args.datasetName)
        # load params
        commonArgs = HYPER_MODEL_MAP[model_name]()['commonParas']
        dataArgs = HYPER_DATASET_MAP[dataset_name]
        dataArgs = dataArgs['aligned'] if (commonArgs['need_data_aligned'] and 'aligned' in dataArgs) else dataArgs['unaligned']
        # integrate all parameters
        self.args = Storage(dict(vars(args),
                            **dataArgs,
                            **commonArgs,
                            **HYPER_MODEL_MAP[model_name]()['datasetParas'][dataset_name],
                            ))
    
    def __datasetCommonParams(self):
        # root_dataset_dir = '/home/HX/workspace/datasets/CHSIMSV2'
        root_dataset_dir = '/data3/HYF/CHSIMSV2'
        tmp = {
            'sims3l':{
                'aligned': {
                    # 'dataPath': os.path.join('simsv3_unsup.pkl'),
                    'dataPath': os.path.join(root_dataset_dir,'unaligned.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 925, 232), # (text, audio, video)
                    'feature_dims': (768, 25, 177), # (text, audio, video)
                    'train_samples': 2722,
                    'train_mix_samples': 12000,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                },
                'unaligned': {
                    # 'dataPath': os.path.join('simsv3_unsup.pkl'),
                    'dataPath': os.path.join(root_dataset_dir,'unaligned.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 925, 232), # (text, audio, video)
                    'feature_dims': (768, 25, 177), # (text, audio, video)
                    'train_samples': 2722,
                    'train_mix_samples': 12000,
                    'num_classes': 3,
                    'language': 'cn',
                    # 'KeyEval': 'Loss',
                    # 'KeyEval': 'F1_score',
                    'KeyEval': 'Mult_acc_2',
                }
            },
        }
        return tmp

    def __V1(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': True,
                'use_bert':True,
                'use_bert_finetune': False,
                'early_stop': 8,
                'weighting_method': 'v1',
            },
            # dataset
            'datasetParas':{
                'sims3l':{
                    'hidden_dims': (64, 16, 16),
                    'post_text_dim': 32,
                    'post_audio_dim': 32,
                    'post_video_dim': 64,
                    'post_fusion_out': 16,
                    'dropouts': (0.1,0.1,0.1),
                    'post_dropouts': (0.3,0.3,0.3,0.3),
                    'batch_size': 32,
                    'M': 1.0,
                    'T': 0.2,
                    'A': 0.8,
                    'V': 0.4,
                    'learning_rate_bert': 5e-4,
                    'learning_rate_audio': 5e-4,
                    'learning_rate_video': 1e-3,
                    'learning_rate_other': 5e-4,
                    'weight_decay_bert': 1e-4,
                    'weight_decay_audio': 0,
                    'weight_decay_video': 0,
                    'weight_decay_other': 5e-4,
                }
            },
        }
        return tmp

    def __V3(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': True,
                'early_stop': 8,
                'fusion_method': 'LMF'
            },
            # dataset
            'datasetParas':{
                'sims3l':{
                    'hidden_dims': (256, 256, 256),
                    'post_text_dim': 256,
                    'post_audio_dim': 128,
                    'post_video_dim': 64,
                    'post_fusion_out': 256,
                    'dropouts': (0.1,0.1,0.1),
                    'post_dropouts': (0.3,0.3,0.3,0.3),
                    'batch_size': 32,
                    'fus_nheads': 4,
                    'fus_layers': 3,
                    'ensemble_depth': 1,
                    'M': 1.0,
                    'T': 1.0,
                    'A': 1.0,
                    'V': 1.0,
                    'learning_rate_bert': 1e-5,
                    'learning_rate_audio': 5e-4,
                    'learning_rate_video': 1e-4,
                    'learning_rate_other': 5e-4,
                    'weight_decay_bert': 1e-5,
                    'weight_decay_audio': 1e-4,
                    'weight_decay_video': 1e-4,
                    'weight_decay_other': 1e-5,
                }
            },
        }
        return tmp
    
    # best hyperparam for layer=6 depth=3
    def __V5(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': True,
                'early_stop': 8,
                'fusion_method': 'LMF'
            },
            # dataset
            'datasetParas':{
                'sims3l':{
                    'hidden_dims': (384, 384, 384),
                    'post_text_dim': 384,
                    'post_audio_dim': 384,
                    'post_video_dim': 384,
                    'post_fusion_out': 384,
                    'dropouts': (0.1,0.1,0.1),
                    'post_dropouts': (0.3,0.3,0.3,0.3),
                    'batch_size': 32,
                    'fus_nheads': 4,
                    'fus_layers': 6,
                    'ensemble_depth': 3,
                    'M': 1.0,
                    'T': 1.0,
                    'A': 1.0,
                    'V': 1.0,
                    'learning_rate_bert': 1e-5,
                    'learning_rate_audio': 5e-5,
                    'learning_rate_video': 5e-5,
                    'learning_rate_other': 5e-5,
                    'weight_decay_bert': 1e-5,
                    'weight_decay_audio': 1e-5,
                    'weight_decay_video': 1e-5,
                    'weight_decay_other': 1e-5,
                }
            },
        }
        return tmp
    
    def __V6(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': True,
                'early_stop': 8,
                'fusion_method': 'trans',
                'weighting_method': 'uw',
                'bert_type': 'roberta',
            },
            # dataset
            'datasetParas':{
                'sims3l':{
                    'hidden_dims': (384, 384, 384),
                    'post_text_dim': 384,
                    'post_audio_dim': 384,
                    'post_video_dim': 384,
                    'post_fusion_out': 384,
                    'dropouts': (0.1,0.1,0.1),
                    'post_dropouts': (0.3,0.3,0.3,0.3),
                    'batch_size': 32,
                    'sub_nheads': 4,
                    'sub_layers': 3,
                    'fus_nheads': 4,
                    'fus_layers': 3,
                    'ensemble_depth': 1,
                    'learning_rate_bert': 1e-5,
                    'learning_rate_audio': 1e-5,
                    'learning_rate_video': 5e-5,
                    'learning_rate_other': 5e-5,
                    'weight_decay_bert': 1e-4,
                    'weight_decay_audio': 1e-4,
                    'weight_decay_video': 1e-4,
                    'weight_decay_other': 1e-4,
                    'ICL_alpha': 0.5,
                    'use_MICL': True,
                    'MICL_alpha': (0.5, 0.5, 0.5),
                }
            },
        }
        return tmp
    
    def __V7(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': True,
                'early_stop': 8,
                'fusion_method': 'trans',
                'weighting_method': 'uw',
                'bert_type': 'roberta',
            },
            # dataset
            'datasetParas':{
                'sims3l':{
                    'hidden_dims': (384, 384, 384),
                    'post_text_dim': 384,
                    'post_audio_dim': 384,
                    'post_video_dim': 384,
                    'post_fusion_out': 384,
                    'dropouts': (0.1,0.1,0.1),
                    'post_dropouts': (0.3,0.3,0.3,0.3),
                    'batch_size': 32,
                    'sub_nheads': 4,
                    'sub_layers': 3,
                    'fus_nheads': 4,
                    'fus_layers': 3,
                    'ensemble_depth': 1,
                    'learning_rate_bert': 1e-5,
                    'learning_rate_audio': 1e-5,
                    'learning_rate_video': 5e-5,
                    'learning_rate_other': 5e-5,
                    'weight_decay_bert': 1e-4,
                    'weight_decay_audio': 1e-4,
                    'weight_decay_video': 1e-4,
                    'weight_decay_other': 1e-4,
                }
            },
        }
        return tmp

    def __V8(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': True,
                'early_stop': 8,
                'fusion_method': 'trans',
                'weighting_method': 'uw',
                'bert_type': 'roberta',
            },
            # dataset
            'datasetParas':{
                'sims3l':{
                    'hidden_dims': (384, 384, 384),
                    'post_text_dim': 384,
                    'post_audio_dim': 384,
                    'post_video_dim': 384,
                    'post_fusion_out': 384,
                    'dropouts': (0.1,0.1,0.1),
                    'post_dropouts': (0.3,0.3,0.3,0.3),
                    'batch_size': 32,
                    'sub_nheads': 4,
                    'sub_layers': 3,
                    'fus_nheads': 4,
                    'fus_layers': 6,
                    'ensemble_depth': 1,
                    'learning_rate_bert': 1e-5,
                    'learning_rate_audio': 1e-5,
                    'learning_rate_video': 5e-5,
                    'learning_rate_other': 5e-5,
                    'weight_decay_bert': 1e-4,
                    'weight_decay_audio': 1e-4,
                    'weight_decay_video': 1e-4,
                    'weight_decay_other': 1e-4,
                    'ICL_alpha': 0.5,
                    'use_MICL': True,
                    'MICL_alpha': (0.5, 0.5, 0.5),
                    # 'use_IBCL': False,
                    # 'IBCL_alpha': (0.1, 0.1, 0.1),
                }
            },
        }
        return tmp

    def get_config(self):
        return self.args