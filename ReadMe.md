# RF-DETR Object Detection TensorRT backend

This is the library for model inference using the TensorRT engine with stream.

## Model

Default: RF-DETR `large` variant, 704x704 square input, 90 COCO foreground
classes (see `cmake/Config.cmake`: `MODEL_NAME = rf_detr_large_704x704`).

RF-DETR requires a **square** input - windowed attention in the DINOv2
backbone has no non-square path. Non-square source images (e.g. KITTI's
1242x375) must be letterbox-padded to a square before calling `infer()` -
see `rf_detr_trt_backend::utils::letterbox_square()` in
`detection_utils.hpp`. `infer()` hard-requires the input already be exactly
`height x width`; it does not resize.

**Preprocessing constants** (fixed, not exposed via CLI flags - override in
`Config` if you ever need different values):

| | Value |
|---|---|
| `mean` (RGB) | `0.485, 0.456, 0.406` |
| `std` (RGB) | `0.229, 0.224, 0.225` |
| `color_order` | RGB (input image itself is BGR8 - see `preprocess_kernel.cu`, which does the swap) |

## Generate the ONNX file

This will generate the FP32 ONNX file in the `onnxs` directory.

\`\`\`bash
# Stock COCO-pretrained weights (large @ 704x704, 90 classes)
python3 ./script/export_rfdetr_to_onnx.py --output-dir ./onnxs

# Fine-tuned checkpoint (variant auto-inferred from the checkpoint)
python3 ./script/export_rfdetr_to_onnx.py \
        --checkpoint runs/my-experiment/checkpoint_best_ema.pth \
        --num-classes 3 \
        --output-dir ./onnxs
\`\`\`

Run `python3 ./script/export_rfdetr_to_onnx.py --help` for the full list of
flags (`--variant`, `--height`/`--width`, `--opset`).

## Convert the FP32 ONNX model to FP16 using ModelOpt AutoCast

Input/output tensors are kept in FP32 for compatibility with the C++
preprocessing pipeline (`keep_io_types=True`).

\`\`\`bash
python3 ./script/convert_onnx_to_fp16.py \
        --input ./onnxs/rf_detr_large_704x704.onnx \
        --output ./onnxs/rf_detr_large_704x704_fp16.onnx
\`\`\`

## Convert to TensorRT engine

Use trtexec to compile the FP16 ONNX to a TensorRT engine. The model is
already in FP16 at the ONNX level.

\`\`\`bash
trtexec --onnx=./onnxs/rf_detr_large_704x704_fp16.onnx \
        --saveEngine=./engines/rf_detr_large_704x704.engine \
        --memPoolSize=workspace:4096 \
        --verbose
\`\`\`

All three steps above run automatically via `cmake/ModelGeneration.cmake`
when building the package (`colcon build` / `cmake --build .`), using the
default (stock COCO) arguments. Re-run the export/convert scripts by hand
with `--checkpoint`/`--num-classes` to build against a fine-tuned model
instead, then re-run `trtexec` (or just rebuild - the custom commands only
re-run when their inputs are newer than their outputs).
