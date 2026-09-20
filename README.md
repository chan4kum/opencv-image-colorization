# Image Colorization

Colorizes black-and-white photos with the pretrained Zhang et al. deep neural network, running through OpenCV's DNN module.

Part of a series of beginner-friendly OpenCV projects.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python download_models.py       # one-time, ~130 MB
python main.py --image bw_photo.jpg --output colorized.jpg
```

Press `q` to quit any live window.

## License

MIT
