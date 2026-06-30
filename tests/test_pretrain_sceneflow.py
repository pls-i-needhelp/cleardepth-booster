"""
tests/test_pretrain_sceneflow.py
==================================
Tests for the Scene Flow (Monkaa) pretraining pipeline
(cleardepth/data/sceneflow_monkaa.py + scripts/pretrain_sceneflow.py).

Covers:
  - PFM reader correctness: vertical flip (bottom-to-top -> top-to-bottom)
    and byte-order/scale-sign convention, verified against synthetic
    files with known values (not by inspection).
  - SceneFlowMonkaaDataset: correct tensor shapes, normalisation,
    disparity sanitisation, disparity rescaling, train/val scene split.
  - Forward + backward pass on synthetic random data: no NaN/Inf.
  - GT downsampling to the model's native 1/4-scale training output,
    exercising pretrain_sceneflow.py's actual downsample_gt() function.

Run with: pytest tests/test_pretrain_sceneflow.py -v
"""

import importlib.util
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from cleardepth.data.sceneflow_monkaa import (
    ORIGINAL_H, ORIGINAL_W, SceneFlowMonkaaDataset, read_pfm,
)
from cleardepth.loss.sequence_loss import SequenceLoss
from cleardepth.models.cleardepth_net import ClearDepthNet


# ===========================================================================
# Helpers
# ===========================================================================

def _write_pfm(path, data, little_endian: bool = True, color: bool = False):
    """Write a PFM file by hand, independent of read_pfm, for cross-checking."""
    data = np.asarray(data, dtype=np.float32)
    h, w = data.shape[:2]
    with open(path, 'wb') as f:
        f.write(b'PF\n' if color else b'Pf\n')
        f.write(f'{w} {h}\n'.encode())
        scale = -1.0 if little_endian else 1.0
        f.write(f'{scale}\n'.encode())
        # PFM stores rows bottom-to-top -> write rows in reverse order
        out = np.flipud(data)
        dtype = '<f4' if little_endian else '>f4'
        f.write(out.astype(dtype).tobytes())


def _build_synthetic_monkaa(root: str, scenes=('sceneA', 'sceneB', 'sceneC', 'sceneD'),
                            n_frames: int = 2, h: int = ORIGINAL_H, w: int = ORIGINAL_W,
                            disp_value=None):
    """
    Build a tiny synthetic Monkaa directory tree.
    If disp_value is given, every disparity map is a constant field of that
    value (deterministic — used to check exact rescaling behaviour).
    Otherwise disparity is random in [0, 50).
    """
    for scene in scenes:
        for cam in ('left', 'right'):
            d = os.path.join(root, 'frames_cleanpass', scene, cam)
            os.makedirs(d, exist_ok=True)
            for i in range(n_frames):
                img = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
                Image.fromarray(img).save(os.path.join(d, f'{i:04d}.png'))
        dd = os.path.join(root, 'disparity', scene, 'left')
        os.makedirs(dd, exist_ok=True)
        for i in range(n_frames):
            if disp_value is not None:
                disp = np.full((h, w), float(disp_value), dtype=np.float32)
            else:
                disp = (np.random.rand(h, w).astype(np.float32) * 50)
            _write_pfm(os.path.join(dd, f'{i:04d}.pfm'), disp)


def _load_pretrain_sceneflow_module():
    """
    Dynamically import scripts/pretrain_sceneflow.py as a module so its
    actual downsample_gt() (and other helpers) can be exercised directly.
    Safe to import: all training logic is gated behind
    `if __name__ == '__main__':`, which is False under this import name.
    """
    script_path = os.path.join(
        os.path.dirname(__file__), '..', 'scripts', 'pretrain_sceneflow.py'
    )
    spec = importlib.util.spec_from_file_location('pretrain_sceneflow', script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# Test Group 1: PFM reader correctness
# ===========================================================================

class TestPFMReader:

    def test_little_endian_roundtrip(self, tmp_path):
        """Known asymmetric pattern survives a write -> read round trip exactly."""
        H, W = 5, 3
        arr = np.array([[(r + 1) * 10 + c for c in range(W)] for r in range(H)],
                       dtype=np.float32)
        path = tmp_path / 'test_le.pfm'
        _write_pfm(str(path), arr, little_endian=True)
        result = read_pfm(str(path))
        np.testing.assert_array_equal(result, arr)

    def test_big_endian_roundtrip(self, tmp_path):
        """Big-endian scale-sign convention (positive scale) is handled correctly."""
        H, W = 5, 3
        arr = np.array([[(r + 1) * 10 + c for c in range(W)] for r in range(H)],
                       dtype=np.float32)
        path = tmp_path / 'test_be.pfm'
        _write_pfm(str(path), arr, little_endian=False)
        result = read_pfm(str(path))
        np.testing.assert_array_equal(result, arr)

    def test_vertical_flip_top_bottom(self, tmp_path):
        """
        PFM stores rows bottom-to-top; read_pfm must flip so row 0 of the
        returned array is the TOP of the image, not the bottom.
        """
        arr = np.zeros((4, 2), dtype=np.float32)
        arr[0] = [100, 101]   # topmost row
        arr[3] = [400, 401]   # bottommost row
        path = tmp_path / 'test_edges.pfm'
        _write_pfm(str(path), arr, little_endian=True)
        result = read_pfm(str(path))
        assert result[0, 0] == 100, "Top row mismatch — vertical flip bug"
        assert result[3, 0] == 400, "Bottom row mismatch — vertical flip bug"

    def test_returns_float32(self, tmp_path):
        arr = np.random.rand(8, 8).astype(np.float32) * 10
        path = tmp_path / 'test_dtype.pfm'
        _write_pfm(str(path), arr)
        result = read_pfm(str(path))
        assert result.dtype == np.float32

    def test_correct_shape(self, tmp_path):
        H, W = 7, 11
        arr = np.random.rand(H, W).astype(np.float32)
        path = tmp_path / 'test_shape.pfm'
        _write_pfm(str(path), arr)
        result = read_pfm(str(path))
        assert result.shape == (H, W)


# ===========================================================================
# Test Group 2: SceneFlowMonkaaDataset
# ===========================================================================

class TestSceneFlowMonkaaDataset:

    @pytest.fixture
    def synthetic_root(self, tmp_path):
        root = str(tmp_path)
        _build_synthetic_monkaa(root)
        return root

    def test_train_val_split_no_leakage(self, synthetic_root):
        train_ds = SceneFlowMonkaaDataset(synthetic_root, split='train',
                                          val_fraction=0.25, seed=1)
        val_ds   = SceneFlowMonkaaDataset(synthetic_root, split='val',
                                          val_fraction=0.25, seed=1)
        train_scenes = {s['scene'] for s in train_ds.samples}
        val_scenes   = {s['scene'] for s in val_ds.samples}
        assert train_scenes.isdisjoint(val_scenes)
        assert len(train_ds) > 0 and len(val_ds) > 0

    def test_sample_shapes(self, synthetic_root):
        ds = SceneFlowMonkaaDataset(synthetic_root, split='train',
                                    height=360, width=720, val_fraction=0.25, seed=1)
        sample = ds[0]
        assert sample['left'].shape == (3, 360, 720)
        assert sample['right'].shape == (3, 360, 720)
        assert sample['disparity'].shape == (1, 360, 720)
        assert isinstance(sample['scene'], str)
        assert isinstance(sample['frame'], str)

    def test_images_normalised_to_minus1_1(self, synthetic_root):
        ds = SceneFlowMonkaaDataset(synthetic_root, split='train', val_fraction=0.25, seed=1)
        sample = ds[0]
        assert sample['left'].min() >= -1.0 - 1e-5
        assert sample['left'].max() <= 1.0 + 1e-5
        assert sample['right'].min() >= -1.0 - 1e-5
        assert sample['right'].max() <= 1.0 + 1e-5

    def test_disparity_sanitised_nonnegative_and_finite(self, synthetic_root):
        """Negative/inf/nan disparity values must be sanitised to 0 by the dataset."""
        root = synthetic_root
        # Inject one scene with inf/nan/negative values
        scene_dir = os.path.join(root, 'disparity', 'sceneA', 'left')
        bad = np.zeros((ORIGINAL_H, ORIGINAL_W), dtype=np.float32)
        bad[0, 0] = np.inf
        bad[0, 1] = np.nan
        bad[0, 2] = -5.0
        _write_pfm(os.path.join(scene_dir, '0000.pfm'), bad)

        # sceneA may land in either split depending on the seed's shuffle —
        # search both so this test doesn't depend on that assignment.
        train_ds = SceneFlowMonkaaDataset(root, split='train', val_fraction=0.25,
                                          seed=1, augment=False)
        val_ds   = SceneFlowMonkaaDataset(root, split='val', val_fraction=0.25,
                                          seed=1, augment=False)
        sample = None
        for ds in (train_ds, val_ds):
            for i, s in enumerate(ds.samples):
                if s['scene'] == 'sceneA' and s['frame'] == '0000':
                    sample = ds[i]
                    break
            if sample is not None:
                break
        assert sample is not None, "Doctored sample not found in either split"

        assert torch.isfinite(sample['disparity']).all(), \
            "inf/nan disparity leaked through un-sanitised"
        assert (sample['disparity'] >= 0).all(), \
            "negative disparity leaked through un-sanitised"

    def test_disparity_rescaled_by_width_ratio(self, tmp_path):
        """
        A constant disparity field of value V at the original resolution
        must become exactly V * (width / ORIGINAL_W) after resize — nearest
        interpolation on a constant field introduces no approximation error,
        so this can be checked exactly rather than statistically.
        """
        root = str(tmp_path)
        _build_synthetic_monkaa(root, scenes=('sceneA',), n_frames=1, disp_value=80.0)
        # val_fraction=1.0 with a single scene guarantees it lands in 'val'
        # (the dataset always holds out at least one scene), avoiding any
        # dependency on the train/val scene-shuffle outcome.
        ds = SceneFlowMonkaaDataset(root, split='val', height=360, width=720,
                                    val_fraction=1.0, seed=1, augment=False)
        assert len(ds) > 0
        sample = ds[0]
        expected = 80.0 * (720 / ORIGINAL_W)
        assert torch.allclose(
            sample['disparity'], torch.full_like(sample['disparity'], expected), atol=1e-3
        ), f"Expected uniform {expected}, got range " \
           f"[{sample['disparity'].min():.3f}, {sample['disparity'].max():.3f}]"

    def test_max_samples_cap(self, synthetic_root):
        ds = SceneFlowMonkaaDataset(synthetic_root, split='train',
                                    max_samples=2, val_fraction=0.25, seed=1)
        assert len(ds) == 2

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SceneFlowMonkaaDataset(str(tmp_path / 'does_not_exist'))


# ===========================================================================
# Test Group 3: Forward + backward pass on synthetic data
# ===========================================================================

class TestForwardBackward:

    def _make_model(self):
        return ClearDepthNet(
            embed_dim=32,
            fuse_out_channels=64,
            depths=[1, 1, 1, 1],
            num_heads=[1, 2, 4, 8],
            reduction_ratios=[8, 4, 2, 1],
            hidden_dim=32,
            n_gru_layers=2,
            n_gru_iters=2,
            corr_levels=2,
            corr_radius=2,
        )

    def test_forward_backward_no_nan(self):
        """A full forward + sequence-loss backward pass must produce no NaN/Inf."""
        model = self._make_model()
        B, H, W = 1, 64, 128
        left  = torch.randn(B, 3, H, W)
        right = torch.randn(B, 3, H, W)
        gt    = torch.rand(B, 1, H, W) * 50 + 1.0

        preds = model(left, right, n_iters=2, test_mode=False)
        assert isinstance(preds, list), \
            "Training mode (test_mode=False) must return a list of predictions"

        _, _, H_q, W_q = preds[0].shape
        disp_scale = W_q / W
        gt_q = F.interpolate(gt, size=(H_q, W_q), mode='nearest') * disp_scale

        loss_fn = SequenceLoss(gamma=0.9, max_disp=192.0 / 4.0)
        loss = loss_fn(preds, gt_q)

        assert torch.isfinite(loss)
        loss.backward()

        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"

    def test_inference_mode_returns_single_tensor(self):
        """
        test_mode=True must return a single full-resolution tensor, not a
        list — per the post-3269bbf contract (plain bilinear x4 upsample
        of the model's native 1/4-scale output).
        """
        model = self._make_model()
        model.eval()
        B, H, W = 1, 64, 128
        left  = torch.randn(B, 3, H, W)
        right = torch.randn(B, 3, H, W)
        with torch.no_grad():
            out = model(left, right, n_iters=2, test_mode=True)
        assert isinstance(out, torch.Tensor), \
            "Inference mode (test_mode=True) must return a single tensor"
        assert out.shape == (B, 1, H, W)


# ===========================================================================
# Test Group 4: GT downsampling to the model's native 1/4-scale output
# ===========================================================================

class TestGTDownsampling:

    def test_downsample_gt_matches_pretrain_sceneflow_logic(self):
        """
        Exercises pretrain_sceneflow.py's actual downsample_gt() function:
        nearest interpolation to the prediction's spatial size, then scale
        disparity VALUES by the same width ratio (1/4 scale -> divide by 4).
        """
        mod = _load_pretrain_sceneflow_module()

        B, H, W = 1, 64, 128
        gt = torch.full((B, 1, H, W), 40.0)
        H_q, W_q = H // 4, W // 4

        gt_q = mod.downsample_gt(gt, H_q, W_q)

        assert gt_q.shape == (B, 1, H_q, W_q)
        # disp_scale = W_q / W = 0.25, so a uniform 40.0 disparity becomes 10.0
        assert torch.allclose(gt_q, torch.full_like(gt_q, 10.0))

    def test_downsampled_gt_matches_model_native_output_scale(self):
        """
        The model's training-mode output (test_mode=False) is natively at
        1/4 scale (confirmed by 3269bbf: full resolution only exists via
        the separate bilinear upsample under test_mode=True). Confirms
        downsample_gt() produces GT at exactly that same resolution, so
        the loss in pretrain_sceneflow.py compares like-for-like.
        """
        model = ClearDepthNet(
            embed_dim=32, fuse_out_channels=64,
            depths=[1, 1, 1, 1], num_heads=[1, 2, 4, 8],
            reduction_ratios=[8, 4, 2, 1], hidden_dim=32,
            n_gru_layers=2, n_gru_iters=2, corr_levels=2, corr_radius=2,
        )
        mod = _load_pretrain_sceneflow_module()

        B, H, W = 1, 64, 128
        left  = torch.randn(B, 3, H, W)
        right = torch.randn(B, 3, H, W)
        preds = model(left, right, n_iters=2, test_mode=False)

        gt = torch.rand(B, 1, H, W) * 50
        _, _, H_q, W_q = preds[0].shape
        gt_q = mod.downsample_gt(gt, H_q, W_q)

        assert gt_q.shape == preds[0].shape, \
            "downsample_gt() output shape must exactly match the model's " \
            "native training-mode (1/4-scale) output shape"
