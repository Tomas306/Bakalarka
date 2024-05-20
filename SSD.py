import cv2
import numpy as np
import tensorflow as tf

# Načítanie modelu
model_path = 'frozen_inference_graph.pb'

# Načítanie TensorFlow grafu
with tf.io.gfile.GFile(model_path, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Vytvorenie grafu
with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name='')

# Vytvorenie TensorFlow session
session = tf.compat.v1.Session(graph=graph)

# Načítanie obrázka
image_path = 'tt/image2023-11-15_054601.jpg'
image = cv2.imread(image_path)
if image is None:
    print(f"Chyba pri načítaní obrázka: {image_path}")
    exit()

(h, w) = image.shape[:2]

# Predspracovanie obrázka
input_tensor = graph.get_tensor_by_name('image_tensor:0')
output_tensors = [
    graph.get_tensor_by_name('detection_boxes:0'),
    graph.get_tensor_by_name('detection_scores:0'),
    graph.get_tensor_by_name('detection_classes:0'),
    graph.get_tensor_by_name('num_detections:0')
]

image_expanded = np.expand_dims(image, axis=0)
(boxes, scores, classes, num) = session.run(output_tensors, feed_dict={input_tensor: image_expanded})

detection_count = 0

# Detekcia osôb
for i in range(int(num[0])):
    if scores[0][i] > 0.01:  # Práh dôveryhodnosti
        class_id = int(classes[0][i])
        if class_id == 1:  # ID triedy pre osoby v COCO dataset
            detection_count += 1
            box = boxes[0][i] * np.array([h, w, h, w])
            (startY, startX, endY, endX) = box.astype("int")
            label = f"Person: {scores[0][i]:.2f}"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


# Zobrazenie počtu úspešných detekcií
print(f"Počet úspešných detekcií: {detection_count}")

# Zobrazenie výsledného obrázka
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()