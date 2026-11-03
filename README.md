# ADOT — Network–Hardware Co-design for CNN Inference on an MCU

ADOT is a desktop tool for running neural networks on small microcontrollers.
It trains a compact CNN that classifies sound or vibration signals (developed
for BLDC motor fault detection), finds the smallest network that still meets
an accuracy target, and checks that it fits the memory and timing limits of a
TI C2000 MCU. The result is quantized and written straight into the firmware
project, which can then be built and flashed. The reference target is the
TMS320F28379D, with parts of the network running on its Control Law
Accelerator (CLA).

## Features

- **Dataset preparation:** turns recordings in class folders into MFCC, log-spectrum or raw features (`adot_dataset.py`)
- **Architecture search:** finds the smallest channel count that reaches the target accuracy, checked across several seeds and signal perturbations
- **Device fit:** memory and inference-time estimates for C2000 parts that have a CLA, with each stage placed on the CPU or the CLA
- **Export:** FLOAT32 / INT16 / INT8 weights and C headers, patched into the firmware submodule

## Repository

| Path | Contents |
| --- | --- |
| `ADOT_Tool.py` | The application (GUI, training, search, export) |
| `adot_dataset.py` | Dataset preparation |
| `CNN_Inference_Edge-AI_with_CLA/` | Submodule: the C2000 firmware |

## Usage

```bash
git clone --recurse-submodules https://github.com/ADOT-Project/Network_Hardware_Co-design_Framework_for_Inference_on_An_MCU.git
pip install torch numpy scipy librosa soundfile PyQt5
python ADOT_Tool.py
```

Ready-to-run Windows builds are available on the
[Releases](https://github.com/ADOT-Project/Network_Hardware_Co-design_Framework_for_Inference_on_An_MCU/releases) page.
