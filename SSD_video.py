import os
import cv2
import numpy as np
import tensorflow as tf
import time

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

# Funkcia na spracovanie videa
def process_video(video_path, output_file, start_time, five_minute_counts, ten_minute_counts, detection_counts, five_minute_mark, ten_minute_mark):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Chyba pri načítaní videa: {video_path}\n")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]

        # Predspracovanie snímky
        input_tensor = graph.get_tensor_by_name('image_tensor:0')
        output_tensors = [
            graph.get_tensor_by_name('detection_boxes:0'),
            graph.get_tensor_by_name('detection_scores:0'),
            graph.get_tensor_by_name('detection_classes:0'),
            graph.get_tensor_by_name('num_detections:0')
        ]

        frame_expanded = np.expand_dims(frame, axis=0)
        (boxes, scores, classes, num) = session.run(output_tensors, feed_dict={input_tensor: frame_expanded})

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
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        detection_counts.append(detection_count)
        elapsed_time = time.time() - start_time

        # Debugovací výstup pre čas a detekcie
        # with open(output_file, 'a', encoding='utf-8') as f:
        #     f.write(f"Uplynulý čas: {elapsed_time:.2f} s, Počet detekcií: {detection_count}\n")

        # Každých 5 minút vypísať priemerný počet detekovaných objektov
        if elapsed_time >= five_minute_mark:
            avg_detection_count = np.mean(detection_counts[-300:])
            five_minute_counts.append(avg_detection_count)
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"Priemerný počet detekcií za posledných 5 minút: {avg_detection_count:.2f}\n")
            print(f"Priemerný počet detekcií za posledných 5 minút: {avg_detection_count:.2f}")
            five_minute_mark += 300  # Aktualizovať časovú značku

        # Každých 10 minút vypísať maximálny počet detekcií
        if elapsed_time >= ten_minute_mark:
            max_detection_count = np.max(detection_counts[-600:])
            ten_minute_counts.append(max_detection_count)
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"Maximálny počet detekcií za posledných 10 minút: {max_detection_count}\n")
            print(f"Maximálny počet detekcií za posledných 10 minút: {max_detection_count}")
            ten_minute_mark += 600  # Aktualizovať časovú značku

        # Zobrazenie výsledného frame-u
        cv2.imshow("Output", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return five_minute_mark, ten_minute_mark

# Prechádzanie priečinka s videami
video_folder = '<cesta_k_súboru>'  # Zmeňte na cestu k priečinku s videami
output_file = '<nazov_výstupneho_suboru>'  # Cesta k výstupnému súboru

# Vymazanie obsahu výstupného súboru na začiatku
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("")

global_start_time = time.time()
detection_counts = []
five_minute_counts = []
ten_minute_counts = []
five_minute_mark = 300
ten_minute_mark = 600

for filename in os.listdir(video_folder):
    if filename.endswith('.avi'):
        video_path = os.path.join(video_folder, filename)
        five_minute_mark, ten_minute_mark = process_video(video_path, output_file, global_start_time, five_minute_counts, ten_minute_counts, detection_counts, five_minute_mark, ten_minute_mark)

# Finálne vypísanie súhrnných štatistík
with open(output_file, 'a', encoding='utf-8') as f:
    if five_minute_counts:
        f.write(f"Priemerné počty detekcií za každých 5 minút: {five_minute_counts}\n")
    if ten_minute_counts:
        f.write(f"Maximálne počty detekcií za každých 10 minút: {ten_minute_counts}\n")