import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data" / "multimodal" / "training_wv3"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "multimodal" / "wv3"

# Hardware specifications
parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")

parser.add_argument("--gpu_id", type=str, default='7')
parser.add_argument('--method', type=str, default='SDL_ms', help='method name')  # Swin_Camba

parser.add_argument('--data_path_train',
                    default=str(DATA_ROOT / "train_wv3.h5"),
                    type=str, help='path of training data')

parser.add_argument("--data_path_test",
                    default=str(DATA_ROOT / "valid_wv3.h5"),
                    type=str, help='path of validation data')

parser.add_argument("--reduced_test_path",
                    default=str(DATA_ROOT / "reduced_examples" / "test_wv3_multiExm1.h5"),
                    type=str, help='path of reduced-resolution test data')
# ======================================================================================================================

# Model specifications
parser.add_argument('--outf', type=str, default=str(OUTPUT_ROOT), help='output directory')

# Training specifications
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument('--band', type=int, default=8, help='the number of HSIs per batch')  # 32
parser.add_argument('--dim', type=int, default=32, help='the number of HSIs per batch')  # 32
parser.add_argument('--batch_size', type=int, default=32, help='the number of HSIs per batch')  # 32
parser.add_argument("--size", default=256, type=int, help='cropped patch size')
parser.add_argument("--max_epoch", type=int, default=500, help='total epoch')
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')

parser.add_argument("--milestones", type=int, default=[50, 100, 150, 200, 250, 300, 350, 400, 450], help='milestones for MultiStepLR')
# [20, 40, 70, 100, 150, 200]，50, 100, 150, 200, 250
parser.add_argument("--gamma", type=float, default=0.9, help='learning rate decay for MultiStepLR')  # 0.6/0.0007
parser.add_argument("--learning_rate", type=float, default=0.0007)  # 0.0008   0.5/0.0004   0.4/0.0003   0.3/0.0002

opt = parser.parse_args()

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False
