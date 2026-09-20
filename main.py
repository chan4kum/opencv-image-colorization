import argparse
import cv2
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--output")
    args = p.parse_args()

    net = cv2.dnn.readNetFromCaffe("models/colorization_deploy_v2.prototxt",
                                   "models/colorization_release_v2.caffemodel")
    pts = np.load("models/pts_in_hull.npy").transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
    net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read {args.image}")
    lab = cv2.cvtColor(img.astype("float32") / 255.0, cv2.COLOR_BGR2LAB)
    L = cv2.resize(lab[:, :, 0], (224, 224)) - 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    out = np.concatenate([lab[:, :, 0:1], ab], axis=2)
    out = np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2BGR), 0, 1)
    out = (out * 255).astype("uint8")
    if args.output:
        cv2.imwrite(args.output, out)
        print("Saved", args.output)
    cv2.imshow("Original | Colorized", np.hstack([img, out]))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
