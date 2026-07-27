# CoSMIC: Hiding Semantics in ISAC

Covert semantic communication over a dual-functional chirp waveform with rectified-flow assisted recovery, evaluated on the View-of-Delft (VoD) automotive dataset.

The sensing output of frame *k-1* (target range / velocity / angle / class + scene image) forms the semantic message of frame *k*. The message is embedded into the rotation of shared Gaussian reference pairs, shaped, and phase-modulated onto a constant-modulus chirp, so the transmit waveform keeps the same envelope and radar response as the sensing-only mode. A per-frame covert projection keeps the KL divergence at the warden below `2*eps^2`. At the receiver, a reliability-guided rectified-flow refiner transports the coarse demodulated latent toward the clean latent in a few steps before semantic decoding.

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

Defaults: 30 epochs, Rayleigh channel with SNR drawn from 0-20 dB, covert budget eps = 0.10, class-balanced sampling over all 13 VoD classes. Checkpoints go to `models/<exp>/best.pt`.

## Test

```bash
python main.py --mode test --snrs 0 5 10 15 20
```

Prints and saves (in `results/<exp>/test_best/`) PSNR, SSIM, and range / velocity / angle RMSE at each SNR, plus the covert KL of every transmitted frame. Add `--demo 2 --demo-snr 15` to also save a couple of closed-loop demo sequences (sensing probe -> covert transmit -> Bob recovery) as png frames + gif.
