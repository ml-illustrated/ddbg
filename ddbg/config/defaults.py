
from fvcore.common.config import CfgNode

_DefC = CfgNode()

_DefC.VERSION = 1

_DefC.project = CfgNode()
_DefC.project.name = ''
_DefC.project.log_dir = './outputs/tb_logs'
_DefC.project.output_dir = './outputs/ddbg_results'

_DefC.model = CfgNode()
_DefC.model.arch = ''
_DefC.model.checkpoint_path = './outputs/ckpts_base'
_DefC.model.embed_layer_name = 'last_linear.'

_DefC.dataset = CfgNode()
_DefC.dataset.name = ''
_DefC.dataset.dataset_path = './data'

_DefC.trainer = CfgNode()
_DefC.trainer.gpus = 1
_DefC.trainer.num_workers = 'auto'
_DefC.trainer.base = CfgNode()
_DefC.trainer.base.epochs = 12
_DefC.trainer.base.batch_size = 64
_DefC.trainer.base.base_lr = 0.001
_DefC.trainer.base.lr_decay_rate = 0.1 # default for resnet50 cifar10

_DefC.trainer.base.optimizer = CfgNode()
_DefC.trainer.base.optimizer.name = 'sgd'
_DefC.trainer.base.optimizer.momentum = 0.9 # default for resnet50 cifar10
_DefC.trainer.base.optimizer.weight_decay = 1e-4 # default for resnet50 cifar10

_DefC.trainer.base.lr_schedule = CfgNode()
_DefC.trainer.base.lr_schedule.steps = 8,-1
_DefC.trainer.base.lr_schedule.cosine = False
# _DefC.trainer.base.lr_schedule.gamma = 0.1


_DefC.data_influence = CfgNode()
_DefC.data_influence.checkpoint_epochs = (4,6,8)
_DefC.data_influence.self_influence_path = './outputs/ddbg_results'
_DefC.data_influence.prop_oppos_path = './outputs/ddbg_results'


_DefC.trainer.embedding = CfgNode()
_DefC.trainer.embedding.name = 'triplets'
_DefC.trainer.embedding.mode = 'rand_negatives'
_DefC.trainer.embedding.pretrained_base_epoch = 6
_DefC.trainer.embedding.self_influence_percentile = 0.9
_DefC.trainer.embedding.base_lr = 0.001
_DefC.trainer.embedding.samples_per_class = 25
_DefC.trainer.embedding.phase_one_epochs = 10
_DefC.trainer.embedding.epochs = 20
_DefC.trainer.embedding.checkpoint_path = './outputs/ckpts_embed'


_DefC.file_names = CfgNode()
_DefC.file_names.save_config = 'ddbg_config.yaml'
_DefC.file_names.project_results = 'ddbg_results.npy'
_DefC.file_names.train_dataset_self_influence_results = 'train_dataset_self_influence_results.npy'
_DefC.file_names.test_dataset_self_influence_results = 'test_dataset_self_influence_results.npy'
_DefC.file_names.train_dataset_prop_oppos_results = 'train_dataset_prop_oppos_results.npy'
_DefC.file_names.test_dataset_prop_oppos_results = 'test_dataset_prop_oppos_results.npy'
_DefC.file_names.train_dataset_mislabel_scores = 'train_dataset_mislabel_scores.npy'
_DefC.file_names.test_dataset_mislabel_scores = 'test_dataset_mislabel_scores.npy'
