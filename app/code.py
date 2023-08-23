import numpy as np
import pickle
import base64
import cv2

Car = {
    0: "Audi",
    1: "Hyundai Creta",
    2: "Mahindra Scorpio",
    3: "Rolls Royce",
    4: "Swift",
    5: "Tata Safari",
    6: "Toyota Innovation",
}


def hog(base64_img):
    img_new = cv2.resize(base64_img, (128, 128), cv2.INTER_AREA)
    win_size = img_new.shape
    cell_size = (8, 8)
    block_size = (16, 16)
    block_stride = (8, 8)
    num_bins = 9

    # Set the parameters of the HOG descriptor using the variables defined above
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, num_bins)

    # Compute the HOG Descriptor for the grayscale image
    hog_descriptor = hog.compute(img_new)
    return hog_descriptor


def readb64(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return img


def Predict_car(model, hog1):
    img = readb64(hog1)
    img_str = hog(img)
    car = model.predict(np.array(img_str).reshape(1, -1))
    return Car[car[0]]
