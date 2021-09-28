import cv2
import argparse
import numpy as np
# from mylearn.cluster import MySpectralClustering
from mylearn.cluster import MyKMeans


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    ap.add_argument("-n", "--n_clusters", type=int, help="Enter n_clusters")
    args = vars(ap.parse_args())

    image = cv2.imread(args["image"])
    (h1, w1) = image.shape[:2]

    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    if np.all(image - image1):
        print("efwefewf")

    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = MyKMeans(n_clusters=args["n_clusters"]).fit(image)

    labels = clt.get_labels() - 1
    segmented_image = clt.get_centers().astype("uint8")[labels]
    segmented_image = segmented_image.reshape((h1, w1, 3))
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_LAB2BGR)
    cv2.imwrite('segmented_image.jpg', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
