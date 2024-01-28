import cv2
import numpy as np
import os


def cut_save(image, output):
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    classes = []
    with open('yolov3.txt', 'r') as f:
        classes = f.read().splitlines()

    img = cv2.imread(image)
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))

            # tao thu muc
            label_folder = os.path.join(output, label)
            if not os.path.exists(label_folder):
                os.makedirs(label_folder)

            # save anh
            anh_moi = img[y:y + h, x:x + w]
            cv2.imwrite(os.path.join(label_folder, f"{label}_{confidence}.jpg"), anh_moi)

        cv2.imshow('image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def save(input, output):
    while True:
        input = input("nhap anh: ")
        output = "Ket_qua"
        cut_save(input, output)

        