"""Downloads the colorization model files into ./models."""
import os
import urllib.request

URLS = {
    "colorization_deploy_v2.prototxt": "https://raw.githubusercontent.com/richzhang/colorization/caffe/colorization/models/colorization_deploy_v2.prototxt",
    "pts_in_hull.npy": "https://github.com/richzhang/colorization/raw/caffe/colorization/resources/pts_in_hull.npy",
    "colorization_release_v2.caffemodel": "https://people.eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel",
}

os.makedirs("models", exist_ok=True)
for name, url in URLS.items():
    path = os.path.join("models", name)
    if not os.path.exists(path):
        print("Downloading", name)
        urllib.request.urlretrieve(url, path)
print("Done.")
