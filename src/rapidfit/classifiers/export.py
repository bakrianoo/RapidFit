"""ONNX export and quantization utilities."""

import json
from pathlib import Path

import torch
import torch.nn as nn


class _ONNXInferenceModel(nn.Module):
    """Lightweight inference wrapper for ONNX export."""

    def __init__(self, encoder, pooler, dropout, head) -> None:
        super().__init__()
        self.encoder = encoder
        self.pooler = pooler
        self.dropout = dropout
        self.head = head

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.pooler(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return self.head(pooled)


def export_to_onnx(
    model: nn.Module,
    task: str,
    output_path: str | Path,
    max_length: int = 128,
    opset_version: int = 17,
    external_data: bool | None = None,
) -> Path:
    """
    Export a single task to ONNX format.

    Args:
        model: The _MultiTaskModel instance.
        task: Task name to export.
        output_path: Directory to save the ONNX file.
        max_length: Maximum sequence length for dummy input.
        opset_version: ONNX opset version.
        external_data: Store weights externally for large models (>2GB).
                       None=auto-detect, True=force, False=disable.

    Returns:
        Path to the exported ONNX file.
    """
    try:
        import onnx
    except ImportError:
        raise ImportError("Install export dependencies: pip install rapidfit[export]")

    if task not in model.task_heads:
        raise ValueError(f"Unknown task: {task}")

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    inference_model = _ONNXInferenceModel(
        encoder=model.encoder,
        pooler=model.pooler,
        dropout=nn.Dropout(0.0),  # No dropout at inference
        head=model.task_heads[task],
    )
    inference_model.eval()

    device = next(model.parameters()).device
    dummy_input_ids = torch.ones(1, max_length, dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long, device=device)

    onnx_path = output_path / f"{task}.onnx"

    if external_data is None:
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        external_data = param_bytes > 2 * 1024**3

    with torch.no_grad():
        torch.onnx.export(
            inference_model,
            (dummy_input_ids, dummy_attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=opset_version,
            do_constant_folding=True,
        )

    if external_data:
        onnx_model = onnx.load(str(onnx_path), load_external_data=False)
        onnx.save(
            onnx_model,
            str(onnx_path),
            save_as_external_data=True,
            location=f"{task}_weights.bin",
            all_tensors_to_one_file=True,
        )
        onnx.checker.check_model(str(onnx_path))
    else:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

    config_path = output_path / f"{task}_config.json"
    config_path.write_text(json.dumps({
        "id2label": {str(k): v for k, v in model.task_id2label[task].items()},
        "label2id": model.task_label2id[task],
    }, indent=2))

    return onnx_path


def quantize_onnx(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """
    Apply INT8 dynamic quantization to an ONNX model.

    Args:
        onnx_path: Path to the ONNX model.
        output_path: Output path. Defaults to <name>_int8.onnx.

    Returns:
        Path to the quantized model.
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        raise ImportError("Install export dependencies: pip install rapidfit[export]")

    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_stem(f"{onnx_path.stem}_int8")
    else:
        output_path = Path(output_path)

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    return output_path
