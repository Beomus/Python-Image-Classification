import numpy as np
import cv2
import time
import os
import imutils

if not os.path.exists("Output"):
    os.mkdir("Output")
img_path = 'Images'

prototxt = 'bvlc_googlenet.prototxt'

model = 'bvlc_googlenet.caffemodel'

labels = 'synset_words.txt'

print("[INFO] Loading labels...")
rows = open(labels).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

print("[INFO] Loading model...")
model = cv2.dnn.readNetFromCaffe(prototxt, model)

for i in os.listdir(img_path):
    # print(i)
    print("[INFO] Loading image...")
    image = cv2.imread(f'{img_path}/{i}')
    image = imutils.resize(image, width=1000)

    # the model expects input shape to be 224x224 pixels
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

    model.setInput(blob)
    start = time.time()
    prediction = model.forward()
    end = time.time()
    print(f"[INFO] Prediction took {round(end-start, 2)}")

    # Taking the top 5 predictions
    idxs = np.argsort(prediction[0])[::-1][:5]

    # printing out the top 5 predictions
    for i, idx in enumerate(idxs): 
        if i == 0:
            # setting text for the prediction with the highest probability
            text = f"Label: {classes[idx]} | Probability: {round(prediction[0][idx] * 100, 2)} | Time: {round(end-start, 2)}"
            cv2.putText(image, text, (5, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        print(f"[INFO] {i} | Label: {classes[idx]} | Probability: {round(prediction[0][idx], 2)}")

    cv2.imshow("Image", image)
    cv2.imwrite(f"Output/{classes[idx]}.png", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
