import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from metrics import ref_evaluate
from utilities.load_test_data import load_h5py, load_h5py1


REPO_ROOT = Path(__file__).resolve().parents[1]

DATASET_CONFIG = {
    "wv3": {
        "test_file": (
            REPO_ROOT
            / "data"
            / "multimodal"
            / "training_wv3"
            / "reduced_examples"
            / "test_wv3_multiExm1.h5"
        ),
        "loader": load_h5py,
    },
    "gf2": {
        "test_file": (
            REPO_ROOT
            / "data"
            / "multimodal"
            / "training_gf2"
            / "reduced_examples"
            / "test_gf2_multiExm1.h5"
        ),
        "loader": load_h5py1,
    },
}


def resolve_repo_path(path_value):
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def test(args):
    config = DATASET_CONFIG[args.dataset]

    if args.file_path is None:
        file_path = config["test_file"]
    else:
        file_path = resolve_repo_path(args.file_path)

    weight_path = resolve_repo_path(args.weight)

    if args.save_dir is None:
        save_dir = (
            REPO_ROOT
            / "outputs"
            / "multimodal"
            / args.dataset
            / "test"
        )
    else:
        save_dir = resolve_repo_path(args.save_dir)

    if not file_path.is_file():
        raise FileNotFoundError(f"Test data not found: {file_path}")

    if not weight_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weight_path}")

    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print(f"Dataset    : {args.dataset.upper()}")
    print(f"Test data  : {file_path}")
    print(f"Checkpoint : {weight_path}")
    print(f"Device     : {device}")
    print(f"Save dir   : {save_dir}")

    model = torch.load(str(weight_path), map_location=device)

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "This test script expects a checkpoint saved as a complete "
            "PyTorch model with torch.save(model, path)."
        )

    model = model.to(device).eval()

    img_lr, _, img_pan, gt = config["loader"](str(file_path))

    image_num = img_lr.shape[0]

    psnr_all = np.zeros(image_num)
    ssim_all = np.zeros(image_num)
    sam_all = np.zeros(image_num)
    ergas_all = np.zeros(image_num)
    scc_all = np.zeros(image_num)
    q_all = np.zeros(image_num)

    with torch.no_grad():
        for k in range(image_num):
            print(f"Processing sample {k + 1}/{image_num}...")

            img_lr_k = img_lr[k:k + 1].to(device)
            img_pan_k = img_pan[k:k + 1].to(device)
            target = gt[k:k + 1].to(device)

            output = model(img_lr_k, img_pan_k)

            if output.shape != target.shape:
                raise RuntimeError(
                    f"Output shape {tuple(output.shape)} does not match "
                    f"target shape {tuple(target.shape)}."
                )

            output_np = (
                output.squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            target_np = (
                target.squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            psnr, ssim, sam, ergas, scc, q = ref_evaluate(
                output_np, target_np
            )

            psnr_all[k] = psnr
            ssim_all[k] = ssim
            sam_all[k] = sam
            ergas_all[k] = ergas
            scc_all[k] = scc
            q_all[k] = q

            print(
                f"PSNR={psnr:.4f}, "
                f"SSIM={ssim:.6f}, "
                f"SAM={sam:.4f}, "
                f"ERGAS={ergas:.4f}"
            )

            save_name = save_dir / f"output_mulExm_{k}.mat"
            sio.savemat(
                str(save_name),
                {
                    "sr": output_np,
                    "tr": target_np,
                },
            )

    print("\n================ Final Results ================")
    print(f"Dataset : {args.dataset.upper()}")
    print(f"PSNR    : {psnr_all.mean():.6f}")
    print(f"SSIM    : {ssim_all.mean():.6f}")
    print(f"SAM     : {sam_all.mean():.6f}")
    print(f"ERGAS   : {ergas_all.mean():.6f}")
    print(f"SCC     : {scc_all.mean():.6f}")
    print(f"Q       : {q_all.mean():.6f}")
    print("================================================")


def main():
    parser = argparse.ArgumentParser(
        description="MambaX multimodal reduced-resolution testing"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["wv3", "gf2"],
        help="dataset to evaluate",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help=(
            "test H5 file; defaults to "
            "<repo>/data/multimodal/<dataset>/reduced_examples/"
        ),
    )
    parser.add_argument(
        "--weight",
        type=str,
        required=True,
        help="path to a trained model checkpoint",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="directory for reconstructed MAT files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="PyTorch device, e.g. cuda, cuda:0, or cpu",
    )

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    main()
