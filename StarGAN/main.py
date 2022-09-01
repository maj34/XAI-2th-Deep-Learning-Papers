import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.  # 데이터로더
    celeba_loader = None
    rafd_loader = None

    if config.dataset in ['CelebA', 'Both']:
        celeba_loader = get_loader(config.celeba_image_dir, config.attr_path, config.selected_attrs,
                                   config.celeba_crop_size, config.image_size, config.batch_size,
                                   'CelebA', config.mode, config.num_workers)
    if config.dataset in ['RaFD', 'Both']:
        rafd_loader = get_loader(config.rafd_image_dir, None, None,
                                 config.rafd_crop_size, config.image_size, config.batch_size,
                                 'RaFD', config.mode, config.num_workers)
    

    # Solver for training and testing StarGAN.
    solver = Solver(celeba_loader, rafd_loader, config)

    # mode에 train을 적으면 solver.py에 정의된 train()함수를 실행하고, test를 적으면 solver.py에 정의된 test()함수를 실행함.
    if config.mode == 'train':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['CelebA', 'RaFD']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
        # c_dim : 데이터셋에서 사용할 특성(attribute)의 수
        # StarGAN에서 기본적으로 CelebA 데이터셋으로부터 Black_Hair, Blond_Hair, Brown_Hair, Male, Young 특성을 사용하기 때문에 default값이 5
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
        # image_size : 모델에 들어갈 이미지 크기(default = 128x128)
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
        # g_conv_dim : Generator 구조에서 첫번째 layer의 filter 수 (논문에 따라 default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
        # d_conv_dim : Discriminator 구조에서 첫번째 layer의 filter 수 (논문에 따라 default = 64)
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
        # g_repeat_num : Generator 구조에서 Residual Block의 수 (논문에 따라 default = 6)
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
        # d_repeat_num : Discriminator 구조에서 Output layer를 제외한 convolution layer의 수 (논문에 따라 default = 6)
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
        # lambda_gp : adversarial loss를 구하는 데에 사용되는 gradient penalty값 (논문에 따라 default = 10)
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='CelebA', choices=['CelebA', 'RaFD', 'Both'])
    parser.add_argument('--batch_size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
        # num_iters : 학습 과정에서 몇 번의 iteration을 돌 것인지
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
        # n_critic : Discriminator가 몇 번 update되었을 때 Generator를 한 번 update 시킬 것인지
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
        # beta1 : n_critic에 사용되는 beta1값
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
        # beta2 : n_critic에 사용되는 beta2값
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for the CelebA dataset',
                        default=['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young'])
        # selected_attrs : CelebA 데이터셋에서 사용할 특성들

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
        # test_iters : 모델 테스트를 위해 학습된 모델을 몇 번때 step에서 가져올 것인지
        # 즉, 모델 학습 시에 model_save_step 인자의 default 값인 10000번째 iteration마다 학습 모델이 저장되는데, 몇 번째 iteration에서 저장된 학습 모델을 가져와 테스트 할 것인지)

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
        # num_workers : 몇 개의 CPU 코어를 할당할 것인지
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
        # mode : train 할 것인지 test 할 것인지
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--celeba_image_dir', type=str, default='data/celeba/images')
        # CelebA 데이터셋을 사용하는 경우 데이터셋이 저장되는 default 경로
    parser.add_argument('--attr_path', type=str, default='data/celeba/list_attr_celeba.txt')
        # CelebA 데이터셋을 사용하는 경우 attribute 정보를 담고 있는 list_attr_celeb.txt파일을 만들어 주어야 하는데, 이 파일이 위치하는 경로
    parser.add_argument('--rafd_image_dir', type=str, default='data/RaFD/train')
        # RaFD 데이터셋을 사용하는 경우 학습용 데이터셋이 저장되는 default 경로(train 폴더 하위에 특성별로 폴더를 만들어 그 안에 이미지 저장)
    parser.add_argument('--log_dir', type=str, default='stargan/logs')
    parser.add_argument('--model_save_dir', type=str, default='stargan/models')
        # model_save_step 인자 값에 해당하는 iteration 수마다 학습 모델이 저장되는 경로
    parser.add_argument('--sample_dir', type=str, default='stargan/samples')
    parser.add_argument('--result_dir', type=str, default='stargan/results')
        # 모델 테스트 결과가 저장되는 경로

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
        # 모델 학습 과정에서 몇 번째 iteration마다 학습 모델을 저장할 것인지
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)