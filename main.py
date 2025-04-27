import os

#os.environ['CUDA_HOME'] = "/usr/local/cuda-11.0"
#os.environ['LD_LIBRARY_PATH'] = "/usr/local/cuda/lib64:"+os.environ['LD_LIBRARY_PATH',""]
#os.environ['CUDA_VISIBLE_DEVICES'] = "MIG-GPU-2c1e2a3f-ecd1-e32a-fce5-ab354c7fcc59/1/0"

#os.environ['CUDA_VISIBLE_DEVICES'] = "MIG-GPU-5f8a90e2-a3ab-fcb0-b33b-ffbb79768e9f/2/0"
os.environ['CUDA_VISIBLE_DEVICES'] = "MIG-GPU-988c1cd4-8d21-84e1-44a0-aa1517e8678b/1/0"


import argparse
from solver_2 import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import numpy
import torch

#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = False

def main(config):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    decay_ratio = 0.5
    decay_epoch = int(config.num_epochs * decay_ratio)
    config.model_path = os.path.join(config.model_path, f'models_lr_{config.lr}_bs_{config.batch_size}')
    config.result_path = os.path.join(config.result_path, f'results_lr_{config.lr}_bs_{config.batch_size}')
    config.num_epochs_decay = decay_epoch

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path, config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)




    print(config)

    train_loader = get_loader(image_path=config.train_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob,
                              image_type = config.image_type,
                              exp_or_sim=config.exp_or_sim,
                              config=config)
    valid_loader = get_loader(image_path=config.valid_path,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.,
                              exp_or_sim=config.exp_or_sim,
                              image_type=config.image_type,
                              config=config)
    test_loader = get_loader(image_path=config.test_path,
                             image_size=config.image_size,
                             batch_size=config.batch_size,
                             num_workers=config.num_workers,
                             mode='test',
                             augmentation_prob=0.,
                             exp_or_sim=config.exp_or_sim,
                             image_type=config.image_type,
                             config=config)

    solver = Solver(config, train_loader, valid_loader, test_loader)

    checkpoint_path = '/raid/crp.dssi/volume_Kubernetes/Benquan/Topological_Network/Network/QR60120180_BCE_20250320/checkpoint_epoch_400.pth______'
    if config.mode == 'train' and os.path.exists(checkpoint_path):
        start_epoch = solver.load_checkpoint(checkpoint_path)
        print(f"Resuming training from epoch {start_epoch + 1}")
        solver.train(start_epoch=start_epoch + 1)
    elif config.mode == 'train':
        solver.train()  # Start training from scratch
    elif config.mode == 'test':
        solver.test(pretrain_path=config.test_pretrained_model_path)
    elif config.mode == 'generate':
        solver.generate_test_result()

import torch.multiprocessing as mp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    image_type = '8bits'  # '8bits' or '16bits'
    dir_name = os.path.dirname(os.path.abspath(__file__))

    #dir_path = f'/raid/crp.dssi/volume_Kubernetes/Benquan/Dataset_QRcode_Simulation/'
    #dir_path = f"/raid/crp.dssi/volume_Kubernetes/Benquan/data_Unet/data_jounral_2023/PracticalDataset/RayleighCluster_{image_type}"
    #dir_path = f"/raid/crp.dssi/volume_Kubernetes/Benquan/data_Unet/data_jounral_2023/PracticalDataset/RayleighCluster_8bits"
    dir_path = f"/raid/crp.dssi/volume_Kubernetes/Benquan/Dataset_QRcode_Experiment/Dataset_QRcode_60nm/10L"
    #save_path = f'/raid/crp.dssi/volume_Kubernetes/Benquan/result_Unet/QRcode_simulation/'



    save_path = f'/raid/crp.dssi/volume_Kubernetes/Benquan/Topological_Network/Image_Result/QRUNET_60_BCE_0424'

    parser.add_argument('--mode', type=str, default='train', help='train | test | generate')
    pretrained_path = "/raid/crp.dssi/volume_Kubernetes/Benquan/Topological_Network/code_benquan/results/topoloss/models_lr_0.0001_bs_16U_NET_0.7focus/U_Net_epoch_75_lr_0.0001_focus_weight_0.7_8bits_U_NET_0.7focus.pkl"
    parser.add_argument('--test_pretrained_model_path', default=pretrained_path, type=str,)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--image_type', type=str, default=image_type)
    parser.add_argument('--special_save_folder_name', type=str, default='github', help='the model would be saved as xxx_{special_save_name}.pkl')
    parser.add_argument('--special_save_name', type=str, default='github', help='the model would be saved as xxx_{special_save_name}.pkl')
    parser.add_argument('--t', type=int, default=3, help='t for Recurrent step of R2U_Net or R2AttU_Net')
    parser.add_argument('--valid_rate', type=float, default=0.1, help="how much ratio of data in the training set would be set as valid data.")
    parser.add_argument('--train_path', type=str,
                        default=os.path.join(dir_path, 'train_valid'))
    parser.add_argument('--valid_path', type=str,
                        default=os.path.join(dir_path, 'train_valid'))
    parser.add_argument('--test_path', type=str,
                        default=os.path.join(dir_path, 'test'))
    #parser.add_argument('--selected_train_valid_fold', type=list, default=['2nanoholes','3nanoholes','4nanoholes','5nanoholes','6nanoholes','7nanoholes','8nanoholes','9nanoholes','10nanoholes'
    #                    ])
    parser.add_argument('--selected_train_valid_fold', type=list, default=['A1','A2','A3'
                        ])
    #parser.add_argument('--selected_train_valid_fold', type=list, default=['A1','A4','A7'
    #                    ])
    #parser.add_argument('--selected_train_valid_fold', type=list, default=['A1','A2','A3',
    #                    ])
    #parser.add_argument('--selected_test_fold', type=list, default=['5nanoholes_90nm_16FOV'])
    parser.add_argument('--selected_test_fold', type=list, default=['A2'])


    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1001)
    parser.add_argument('--num_epochs_decay', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--beta1', type=float, default=0.9)  # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.995)  # momentum2 in Adam
    parser.add_argument('--wd', type=float, default=0)  # momentum2 in Adam
    parser.add_argument('--augmentation_prob', type=float, default=0.4)
    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)
    parser.add_argument('--focus_weight', type=float, default=1, help="new_loss = bce_loss + focus_weight * focus_loss")
    parser.add_argument('--focus_beta', type=float, default=0.5, help='beta for f_beta loss')
    parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net')
    #parser.add_argument('--model_type', type=str, default='U_Net', help='U_Net')
    parser.add_argument('--model_path', type=str, default=save_path)
    parser.add_argument('--result_path', type=str, default=save_path)
    parser.add_argument('--exp_or_sim', type=str, default=None)
    parser.add_argument('--rotate', type=bool, default=True)
    parser.add_argument('--center_crop', type=bool, default=False)
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--cuda_idx', type=int, default=0)

    config = parser.parse_args()
    if config.mode == 'train':
        try:
            main(config)
        except Exception as e:
            raise e
    elif config.mode == 'test':
        #test_fold = ['QRcode']
        


        #for fold in test_fold:
            #config.selected_test_fold = [fold]
            #config.result_path = save_path
            main(config)
