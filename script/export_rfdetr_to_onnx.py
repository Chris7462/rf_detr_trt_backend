#!/usr/bin/env python3
"""
Export a pre-trained (or fine-tuned) RF-DETR model to ONNX format for
TensorRT/C++ inference.

Uses the official roboflow/rf-detr Python package. Defaults match the
`large` variant at 704x704 - the confirmed choice for the rf_detr_trt_backend
port (see cmake/Config.cmake: MODEL_NAME = rf_detr_large_704x704). No
override flags are passed by cmake/ModelGeneration.cmake, so these defaults
are what actually gets built unless you invoke this script by hand with
different arguments.

RF-DETR requires a SQUARE input (windowed attention in the DINOv2 backbone
has no non-square path) - --height and --width must be equal; this script
enforces that rather than silently ignoring a mismatch.

Usage:
    # Stock COCO-pretrained weights (default: large @ 704x704, 90 classes)
    python export_rfdetr_to_onnx.py --output-dir onnxs

    # Fine-tuned KITTI checkpoint (variant auto-inferred from the checkpoint)
    python export_rfdetr_to_onnx.py \\
        --checkpoint runs/kitti-large/checkpoint_best_ema.pth \\
        --num-classes 3 \\
        --output-dir onnxs
"""

import argparse
import json
import os
from pathlib import Path


_VARIANT_CLASSES = {
    'nano': 'RFDETRNano',
    'small': 'RFDETRSmall',
    'medium': 'RFDETRMedium',
    'base': 'RFDETRBase',
    'large': 'RFDETRLarge',
}


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', type=str, default=None,
                    help='Path to a fine-tuned .pth checkpoint (e.g. '
                         'checkpoint_best_ema.pth). If omitted, exports the '
                         'stock COCO-pretrained weights for --variant.')
    ap.add_argument('--variant', type=str, default='large',
                    choices=sorted(_VARIANT_CLASSES.keys()),
                    help='RF-DETR variant. Ignored if --checkpoint is given - '
                         'the variant is auto-inferred from the checkpoint '
                         'itself. Segmentation variants are out of scope for '
                         'this detection-only backend.')
    ap.add_argument('--height', type=int, default=704, help='Square input side (height)')
    ap.add_argument('--width', type=int, default=704, help='Square input side (width)')
    ap.add_argument('--num-classes', type=int, default=90,
                    help='Number of foreground classes (excluding background). '
                         'Stock COCO checkpoints use 90; override to match your '
                         'fine-tuned checkpoint\'s class count (e.g. 3 for a '
                         'Car/Pedestrian/Cyclist KITTI fine-tune).')
    ap.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    ap.add_argument('--output-dir', type=str, default='onnxs',
                    help='Directory to write the .onnx file into')
    return ap.parse_args()


def load_model(args):
    if args.checkpoint:
        # from_checkpoint() infers the correct variant subclass from the
        # checkpoint itself - no need to guess/match args.variant by hand.
        from rfdetr import RFDETR
        print(f'Loading fine-tuned checkpoint: {args.checkpoint}')
        return RFDETR.from_checkpoint(
            args.checkpoint, num_classes=args.num_classes, resolution=args.height)

    import rfdetr
    cls_name = _VARIANT_CLASSES[args.variant]
    model_cls = getattr(rfdetr, cls_name)
    print(f'Loading stock COCO-pretrained {cls_name} weights...')
    return model_cls(num_classes=args.num_classes, resolution=args.height)


def main():
    args = parse_args()

    if args.height != args.width:
        raise ValueError(
            f'RF-DETR requires a square input (height must equal width), got '
            f'height={args.height}, width={args.width}. Windowed attention in '
            f'the DINOv2 backbone has no non-square path - see the port\'s '
            f'design notes.')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'=== Exporting RF-DETR ({args.variant if not args.checkpoint else "from checkpoint"}) '
          f'for input size: {args.height}x{args.width}, {args.num_classes} classes ===')

    model = load_model(args)

    print('Exporting to ONNX...')
    model.export(
        output_dir=str(out_dir),
        shape=(args.height, args.width),
        opset_version=args.opset,
    )

    # model.export() always writes "inference_model.onnx" into output_dir -
    # rename to match cmake/Config.cmake's ONNX_FILE = "${MODEL_NAME}.onnx".
    basename = f'rf_detr_{args.variant}_{args.height}x{args.width}'
    dst = out_dir / f'{basename}.onnx'

    known_candidates = [
        out_dir / 'inference_model.onnx',
        out_dir / f'rfdetr-{args.variant}.onnx',
    ]
    src = next((p for p in known_candidates if p.exists()), None)

    if src is None:
        # Fallback: newest .onnx file in out_dir that isn't already our
        # target name (avoids picking up a stale file from a prior run).
        candidates = sorted(
            (p for p in out_dir.glob('*.onnx') if p != dst),
            key=lambda p: p.stat().st_mtime, reverse=True)
        src = candidates[0] if candidates else None

    if src is None or not src.exists():
        raise RuntimeError(
            f'export() reported success but no .onnx file was found in '
            f'{out_dir} - checked {[str(p) for p in known_candidates]} and '
            f'the newest-file fallback. Run `ls {out_dir}` to see what the '
            f'installed rfdetr package actually wrote, and update this '
            f'script\'s known_candidates list to match.')
    src.rename(dst)
    print(f'Renamed {src.name} -> {dst.name}')
    print(f'ONNX model saved to: {dst}')

    # Validate the exported graph
    try:
        import onnx
        onnx_model = onnx.load(str(dst))
        onnx.checker.check_model(onnx_model)
        print('ONNX model validation passed')

        output_names = {o.name for o in onnx_model.graph.output}
        if 'masks' in output_names:
            print('WARNING: exported graph has a "masks" output - this looks '
                  'like a segmentation checkpoint. rf_detr_trt_backend is '
                  'detection-only and expects exactly two outputs (dets, labels).')
        missing = {'dets', 'labels'} - output_names
        if missing:
            print(f'WARNING: expected output(s) {missing} not found in the '
                  f'exported graph (found: {sorted(output_names)}). '
                  f'rf_detr_trt_backend\'s find_tensor_names() falls back to '
                  f'positional matching, but exact names are safer.')
    except ImportError:
        print('onnx package not available - skipping model validation (pip install onnx)')
    except Exception as e:
        print(f'ONNX model validation failed: {e}')

    # Reference sidecar for humans only - NOT read by the C++ backend.
    # rf_detr_trt_backend uses Config (passed by the caller), not a JSON
    # sidecar, by design.
    meta = {
        'variant': args.variant,
        'input_h': args.height,
        'input_w': args.width,
        'num_classes': args.num_classes,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'color_order': 'RGB',
        'checkpoint': args.checkpoint or 'stock-coco-pretrained',
    }
    meta_path = out_dir / f'{basename}.reference.json'
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    print(f'Wrote reference metadata (informational only): {meta_path}')

    print('ONNX export completed.')


if __name__ == '__main__':
    main()
