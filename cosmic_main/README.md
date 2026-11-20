# CoSMIC: Hiding Semantics in ISAC

Covert semantic communication over a dual-functional chirp waveform with rectified-flow assisted recovery, evaluated on the View-of-Delft (VoD) automotive dataset.

The sensing output of frame *k-1* (target range / velocity / angle / class + scene image) forms the semantic message of frame *k*. The message is embedded into the rotation of shared Gaussian reference pairs, shaped, and phase-modulated onto a constant-modulus chirp, so the transmit waveform keeps the same envelope and radar response as the sensing-only mode. The covertness budget `2*eps^2` is shared evenly over a monitoring window of `K_W` frames, and a per-frame covert projection keeps the KL cost of every frame below `2*eps^2 / K_W`. At the receiver, a reliability-guided rectified flow on the circle transports the coarse pair angles toward the clean ones in two Euler steps before semantic decoding.

## Layout

| file | what it does |
| --- | --- |
| `modules.py` | waveform + covert projection + channel + receiver + all networks + radar matched filter |
| `dataset.py` | VoD loader with the sense-then-transmit frame pairing |
| `train.py` | end-to-end training and evaluation |
| `test.py` | PSNR / SSIM / sensing RMSE vs SNR, optional demo sequences |
| `main.py` | config + entry point |

## Requirements

- Python 3.10+, PyTorch 2.x with CUDA, numpy, pillow, matplotlib, tqdm
- [View-of-Delft dataset](https://github.com/tudelft-iv/view-of-delft-dataset) under `dataset/VoD/view_of_delft_PUBLIC` (or pass `--vod-root`)

## Train

```bash
python main.py --mode train
```

Defaults: 30 epochs, Rayleigh channel with SNR drawn from 0-20 dB, covertness budget eps = 0.10 over a window of K_W = 1000 frames (`--eps`, `--window`), class-balanced sampling over all 13 VoD classes. The model is validated after every epoch and checkpoints go to `models/<exp>/best.pt`.

## Test

```bash
python main.py --mode test --snrs 0 5 10 15 20
```

Prints and saves (in `results/<exp>/test_best/`) PSNR, SSIM, and range / velocity / angle RMSE at each SNR, plus the covert KL of every transmitted frame. `--eps` and `--window` evaluate the checkpoint under another budget. Add `--demo 2 --demo-snr 15` to also save a couple of closed-loop demo sequences (sensing probe -> covert transmit -> Bob recovery) as png frames + gif.
