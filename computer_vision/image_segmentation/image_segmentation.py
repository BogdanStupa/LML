import cv2
from mylearn.cluster import MyKMeans
# from mylearn.cluster import MySpectralClustering

if __name__ == "__main__":
    image = cv2.imread('fruits.jpg')
    (h1, w1) = image.shape[:2]

    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = MyKMeans(n_clusters=3).fit(image)

    labels = clt.get_labels() - 1
    quant = clt.get_centers().astype("uint8")[labels]
    print(clt.get_centers())

    quant = quant.reshape((h1, w1, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    cv2.imwrite('generated.jpg', quant)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
