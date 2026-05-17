# MPS Examples

Activate the Mac environment first:

```bash
conda activate sana-mps
export PYTORCH_ENABLE_MPS_FALLBACK=1
export DISABLE_XFORMERS=1
```

Generate an image using the local SANA1.5 image model:

```bash
python examples/mps_image.py
```

This defaults to:

```text
/Users/frankfacundo/Models/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers
```

Use the lighter local 600M 512px image model:

```bash
python examples/mps_image.py \
  --model-path /Users/frankfacundo/Models/Efficient-Large-Model/Sana_600M_512px_diffusers \
  --height 512 \
  --width 512 \
  --output sana_600m_512.png
```

Use a custom prompt:

```bash
python examples/mps_image.py \
  --prompt 'a small glass cabin in a snowy pine forest at sunrise' \
  --output output.png
```

Generate a short video using the local Sana-Video model:

```bash
python examples/mps_video.py
```

This defaults to:

```text
/Users/frankfacundo/Models/Efficient-Large-Model/SANA-Video_2B_480p_diffusers
```

The local SANA1.5 folder is an image model, so use `examples/mps_image.py` for it.
