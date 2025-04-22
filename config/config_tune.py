import os
import random
import argparse

from utils.functions import Storage


class ConfigTune():
    def __init__(self, args):
        # global parameters for running
        self.globalArgs = args
        # hyper parameters for models
        HYPER_MODEL_MAP = {
            # multi-task
            'v1': self.__V1,
            'v1_semi': self.__V1_Semi,
            'v2': self.__V1,
            'v3': self.__V3,
            'v5': self.__V5,
            'v6': self.__V6,
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
                            **HYPER_MODEL_MAP[model_name]()['debugParas'],
                            ))
    
    def __datasetCommonParams(self):
        # root_dataset_dir = '/home/HYF/workspace/datasets/CHSIMSV2'
        root_dataset_dir = '/data3/HYF/CHSIMSV2'
        tmp = {
            'sims3':{
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir,'SimsLargeV1.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (41, 925, 232), # (text, audio, video)
                    'feature_dims': (768, 25, 177), # (text, audio, video)
                    'train_samples': 2722,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                }
            },
           'sims3l':{
                'aligned': {
                    'dataPath': os.path.join(root_dataset_dir,'unaligned.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 50, 50), # (text, audio, video)
                    'feature_dims': (768, 25, 177), # (text, audio, video)
                    'train_samples': 2722,
                    'train_mix_samples': 12884,
                    'num_classes': 3,
                    'language': 'cn',
                    'KeyEval': 'Loss',
                },
                'unaligned': {
                    'dataPath': os.path.join(root_dataset_dir,'unaligned.pkl'),
                    # (batch_size, seq_lens, feature_dim)
                    'seq_lens': (50, 925, 232), # (text, audio, video)
                    'feature_dims': (768, 25, 177), # (text, audio, video)
                    'train_samples': 2722,
                    'train_mix_samples': 12884,
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
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': False,
                'early_stop': 8
            },
            'debugParas':{
                'd_paras': ['hidden_dims','post_fusion_in','post_fusion_out',\
                            'dropouts','post_dropouts','batch_size',\
                            'fus_nheads','fus_layers','fus_relu_dropout','fus_embed_dropout','fus_res_dropout','fus_attn_dropout','fus_position_embedding',\
                            'M', 'T', 'A', 'V', \
                            'learning_rate_bert', 'learning_rate_audio','learning_rate_video', 'learning_rate_other',\
                            'weight_decay_bert', 'weight_decay_audio', 'weight_decay_video', 'weight_decay_other'],
                
                'hidden_dims': random.choice([(256,32,64),(256,64,64),(128,32,128), (128,32,64), (128,32,32),(64,32,32),(64,16,32), (64,16,16)]),
                'post_fusion_dim': random.choice([16,32,64]),
                'post_text_dim': random.choice([8,16,32,64]),
                'post_audio_dim': random.choice([8,16,32]),
                'post_video_dim': random.choice([8,16,32,64]),
                'post_fusion_in': random.choice([16,32,64,128]),
                'post_fusion_out': random.choice([8,16,32]),
                'fus_nheads':random.choice([2,4,8]),
                'fus_layers':random.choice([3,4,5,6]),
                'fus_relu_dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                'fus_embed_dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                'fus_res_dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                'fus_attn_dropout': random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                'fus_position_embedding': random.choice([True, False]),
                'dropouts': random.choice([(0.1,0.1,0.1),(0.2,0.2,0.2)]),
                'post_dropouts': random.choice([(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2)]),
                # # ref Original Paper
                'batch_size': random.choice([32,64]),
                # ref Original Paper
                'M':random.choice([0.2,0.4,0.6,0.8,1]),
                'T':random.choice([0.2,0.4,0.6,0.8,1]),
                'A':random.choice([0.2,0.4,0.6,0.8,1]),
                'V':random.choice([0.2,0.4,0.6,0.8,1]),
                'learning_rate_bert': random.choice([5e-4, 5e-5]),
                'learning_rate_audio': random.choice([5e-4, 1e-3]),
                'learning_rate_video': random.choice([5e-4, 1e-3]),
                'learning_rate_other': random.choice([5e-4, 1e-3]),
                'weight_decay_bert': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_audio': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_video': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_other': random.choice([0, 5e-4, 1e-5]),
            }
        }
        return tmp

    def __V1_Semi(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_sampling': False,
                'need_sampling_fix': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': False,
                'early_stop': 8
            },
            'debugParas':{
                'd_paras': ['hidden_dims','post_fusion_dim','post_text_dim','post_audio_dim',\
                            'post_video_dim','dropouts','post_dropouts','batch_size',\
                            'M', 'T', 'A', 'V', \
                            'learning_rate_bert', 'learning_rate_audio','learning_rate_video', 'learning_rate_other',\
                            'weight_decay_bert', 'weight_decay_audio', 'weight_decay_video', 'weight_decay_other'],
                'hidden_dims': random.choice([(128,32,128),(64,32,64),(64,16,32)]),
                'post_fusion_dim': random.choice([16,32,64]),
                'post_text_dim': random.choice([8,16,32,64]),
                'post_audio_dim': random.choice([8,16,32]),
                'post_video_dim': random.choice([8,16,32,64]),
                'dropouts': random.choice([(0.1,0.1,0.1),(0.2,0.2,0.2),(0.3,0.3,0.3)]),
                'post_dropouts': random.choice([(0.3,0.3,0.3,0.3),(0.2,0.2,0.2,0.2),(0.4,0.4,0.4,0.4)]),
                # ref Original Paper
                'batch_size': random.choice([32, 64, 128]),
                'M':random.choice([0.2,0.4,0.6,0.8,1]),
                'T':random.choice([0.2,0.4,0.6,0.8,1]),
                'A':random.choice([0.2,0.4,0.6,0.8,1]),
                'V':random.choice([0.2,0.4,0.6,0.8,1]),
                'learning_rate_bert': 5e-5,
                'learning_rate_audio': random.choice([5e-4, 1e-3]),
                'learning_rate_video': random.choice([5e-4, 1e-3]),
                'learning_rate_other': random.choice([5e-4, 1e-3]),
                'weight_decay_bert': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_audio': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_video': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_other': random.choice([0, 5e-4, 1e-5]),
            }
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
                'early_stop': 8
            },
            'debugParas':{
                'd_paras': ['hidden_dims',
                            'post_text_dim','post_audio_dim','post_video_dim','post_fusion_out',\
                            'fus_nheads','fus_layers','ensemble_depth',\
                            'dropouts','post_dropouts','batch_size',\
                            'M', 'T', 'A', 'V', \
                            'learning_rate_bert', 'learning_rate_audio','learning_rate_video', 'learning_rate_other',\
                            'weight_decay_bert', 'weight_decay_audio', 'weight_decay_video', 'weight_decay_other'],
                
                # 'hidden_dims': random.choice([(256,32,64),(256,64,64),(128,32,128), (128,32,64), (128,128,128)]),
                'hidden_dims': random.choice([(256,256,256)]),
                'post_text_dim': random.choice([64,128,256]),
                'post_audio_dim': random.choice([64,128]),
                'post_video_dim': random.choice([64,128]),
                'post_fusion_out': random.choice([128,256]),
                'ensemble_depth': random.choice([1]),
                'fus_nheads': random.choice([4]),
                'fus_layers': random.choice([3]),
                'dropouts': random.choice([(0.1,0.1,0.1)]),
                'post_dropouts': random.choice([(0.3,0.3,0.3,0.3)]),
                # # ref Original Paper
                'batch_size': random.choice([32]),
                # ref Original Paper
                # 'M':random.choice([0.2,0.4,0.6,0.8,1]),
                # 'T':random.choice([0.2,0.4,0.6,0.8,1]),
                # 'A':random.choice([0.2,0.4,0.6,0.8,1]),
                # 'V':random.choice([0.2,0.4,0.6,0.8,1]),
                'M':random.choice([1]),
                'T':random.choice([1]),
                'A':random.choice([1]),
                'V':random.choice([1]),
                'learning_rate_bert': random.choice([1e-5]),
                'learning_rate_audio': random.choice([5e-5, 5e-4, 1e-4]),
                'learning_rate_video': random.choice([5e-5, 5e-4, 1e-4]),
                'learning_rate_other': random.choice([5e-5, 5e-4, 1e-4]),
                'weight_decay_bert': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_audio': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_video': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_other': random.choice([0, 1e-4, 1e-5]),
            }
        }
        return tmp

    def __V4(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': True,
                'early_stop': 8
            },
            'debugParas':{
                'd_paras': ['hidden_dims',
                            'post_text_dim','post_audio_dim','post_video_dim','post_fusion_out',\
                            'fus_nheads','fus_layers','fus_position_embedding','ensemble_depth',\
                            'dropouts','post_dropouts','batch_size',\
                            'M', 'T', 'A', 'V', \
                            'learning_rate_bert', 'learning_rate_audio','learning_rate_video', 'learning_rate_other',\
                            'weight_decay_bert', 'weight_decay_audio', 'weight_decay_video', 'weight_decay_other'],
                
                'hidden_dims': random.choice([(256,256,256)]),
                'post_text_dim': random.choice([128]),
                'post_audio_dim': random.choice([64]),
                'post_video_dim': random.choice([64]),
                'post_fusion_out': random.choice([256]),
                'fus_nheads':random.choice([2,4,8]),
                'fus_layers':random.choice([3,4,5,6]),
                'ensemble_depth':random.choice([1,2,3]),
                'fus_position_embedding': random.choice([True]),
                'dropouts': random.choice([(0.1,0.1,0.1)]),
                'post_dropouts': random.choice([(0.3,0.3,0.3,0.3)]),
                # # ref Original Paper
                'batch_size': random.choice([32]),
                'M':random.choice([1]),
                'T':random.choice([1]),
                'A':random.choice([1]),
                'V':random.choice([1]),
                'learning_rate_bert': random.choice([1e-5]),
                'learning_rate_audio': random.choice([1e-4, 5e-4]),
                'learning_rate_video': random.choice([1e-4]),
                'learning_rate_other': random.choice([5e-5]),
                'weight_decay_bert': random.choice([1e-5]),
                'weight_decay_audio': random.choice([1e-5]),
                'weight_decay_video': random.choice([0]),
                'weight_decay_other': random.choice([0]),
            }
        }
        return tmp
    
    def __V5(self):
        tmp = {
            'commonParas':{
                'need_data_aligned': False,
                'need_model_aligned': False,
                'need_normalized': False,
                'use_bert':True,
                'use_bert_finetune': True,
                'early_stop': 8
            },
            'debugParas':{
                'd_paras': ['hidden_dims',
                            'post_text_dim','post_audio_dim','post_video_dim','post_fusion_out',\
                            'fus_nheads','fus_layers','fus_position_embedding','ensemble_depth',\
                            'dropouts','post_dropouts','batch_size',\
                            'M', 'T', 'A', 'V', \
                            'learning_rate_bert', 'learning_rate_audio','learning_rate_video', 'learning_rate_other',\
                            'weight_decay_bert', 'weight_decay_audio', 'weight_decay_video', 'weight_decay_other'],
                
                'hidden_dims': random.choice([(256,256,256)]),
                'post_text_dim': random.choice([64,128,256]),
                'post_audio_dim': random.choice([64,128]),
                'post_video_dim': random.choice([64,128]),
                'post_fusion_out': random.choice([128,256]),
                'fus_nheads':random.choice([4]),
                'fus_layers':random.choice([6]),
                'ensemble_depth':random.choice([2,3]),
                'fus_position_embedding': random.choice([True]),
                'dropouts': random.choice([(0.1,0.1,0.1)]),
                'post_dropouts': random.choice([(0.3,0.3,0.3,0.3)]),
                # # ref Original Paper
                'batch_size': random.choice([32]),
                'M':random.choice([1]),
                'T':random.choice([1]),
                'A':random.choice([1]),
                'V':random.choice([1]),
                'learning_rate_bert': random.choice([1e-5]),
                'learning_rate_audio': random.choice([5e-5, 5e-4, 1e-4]),
                'learning_rate_video': random.choice([5e-5, 5e-4, 1e-4]),
                'learning_rate_other': random.choice([5e-5, 5e-4, 1e-4]),
                'weight_decay_bert': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_audio': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_video': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_other': random.choice([0, 1e-4, 1e-5]),
            }
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
                'fusion_method': 'MMT'

            },
            'debugParas':{
                'd_paras': ['hidden_dims',
                            'post_text_dim','post_audio_dim','post_video_dim','post_fusion_out',\
                            'sub_nheads','sub_layers','fus_nheads','fus_layers','ensemble_depth',\
                            'dropouts','post_dropouts','batch_size',\
                            'learning_rate_bert', 'learning_rate_audio','learning_rate_video', 'learning_rate_other',\
                            'weight_decay_bert', 'weight_decay_audio', 'weight_decay_video', 'weight_decay_other'],
                
                'hidden_dims': random.choice([(768, 768, 768)]),
                'post_text_dim': random.choice([384]),
                'post_audio_dim': random.choice([384]),
                'post_video_dim': random.choice([384]),
                'post_fusion_out': random.choice([768]),
                'sub_nheads': 4,
                'sub_layers': 3,
                'fus_nheads':random.choice([4]),
                'fus_layers':random.choice([3]),
                'ensemble_depth':random.choice([1]),
                'dropouts': random.choice([(0.1,0.1,0.1)]),
                'post_dropouts': random.choice([(0.3,0.3,0.3,0.3)]),
                # # ref Original Paper
                'batch_size': random.choice([24]),
                'learning_rate_bert': random.choice([1e-5]),
                'learning_rate_audio': random.choice([1e-5, 5e-5]),
                'learning_rate_video': random.choice([1e-5, 5e-5]),
                'learning_rate_other': random.choice([1e-5, 5e-5]),
                'weight_decay_bert': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_audio': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_video': random.choice([0, 1e-4, 1e-5]),
                'weight_decay_other': random.choice([0, 1e-4, 1e-5]),
            }
        }
        return tmp
    
    
    
    
    
    def get_config(self):
        return self.args