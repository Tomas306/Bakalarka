import cv2
import numpy as np
import os
import time
import datetime



# Inicializácia HOG detektora
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Načítanie masky
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)  # Predpokladáme, že maska je čiernobiela

# Funkcia na zápis výstupu do textového súboru
def write_output(output):
    with open("20240514_output.txt", "a") as file:
        file.write(output + "\n")
# Cesta k priečinku s video súbormi
video_folder_path = 'Videa/20240513/ja'

# Funkcia na získanie zoznamu video súborov v priečinku
def get_video_files(folder_path):
    video_files = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.avi') or file_name.endswith('.mp4'):
            video_files.append(os.path.join(folder_path, file_name))
    return video_files

# Získanie zoznamu video súborov
video_files = get_video_files(video_folder_path)


# Skontrolovanie, či sa našli nejaké video súbory
if not video_files:
    print("Error: No video files found in the folder.")
    exit()

# Začiatočný čas
global_start_time = time.time()
global_last_5min_time = global_start_time  # Začiatočný čas pre 5 minútový interval
global_last_10min_time = global_start_time  # Začiatočný čas pre 10 minútový interval

all_detections = []  # Zoznam na uchovanie počtu detekcií pre všetky videá

videocounter = 0

# Spracovanie každého video súboru postupne
for video_path in video_files:
    videocounter = videocounter + 1
    # Debug výstup pre zistenie či sa video načítalo správne
    print(f"Processing video: {video_path}, toto je {videocounter} video")

    # Otvorenie video súboru
    video_capture = cv2.VideoCapture(video_path)

    # Skontrolovanie, či sa video otvorilo správne
    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}.")
        continue

    # Načítanie prvého snímku na získanie rozmerov
    ret, frame = video_capture.read()
    if not ret:
        print(f"Error: Could not read frame from video {video_path}.")
        continue

    # Zmena veľkosti masky na rozmer snímky
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    video_detections = []  # Zoznam na uchovanie počtu detekcií pre aktuálne video

    # Vrátenie video streamu na začiatok
    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Získanie aktuálneho času
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Deaktivácia detekcie v určených oblastiach pomocou masky
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask_resized)
        # Detekcia osôb v snímke
        boxes, weights = hog.detectMultiScale(masked_frame, winStride=(2,2), padding=(8, 8), scale=1.05, hitThreshold=0)

        # Vykreslenie obdĺžnikov okolo detegovaných osôb
        for (x, y, w, h) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        video_detections.append(len(boxes))
        all_detections.append(len(boxes))

        # Zobrazenie výsledného snímku
        cv2.imshow('Detekcia osôb pomocou HOG', frame)

        # Ukončenie slučky pri stlačení klávesy 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        # Funkcia na zápis výstupu do textového súboru
        def write_output(output):
            with open("20240514_output.txt", "a") as file:
                file.write(output + "\n")
        current_time_seconds = time.time()  # Aktuálny čas v sekundách
        # Každých 5 minút
        if current_time_seconds - global_last_5min_time >= 300:  # Ak ubehlo 5 minút
            average_detections = sum(all_detections) / len(all_detections) if all_detections else 0
            output = f"Dátum a čas spracovania: {current_time}\n"
            output += f"Priemerný počet detekcií osôb za 5 minút: {average_detections:.2f}\n"
            write_output(output)
            global_last_5min_time = current_time_seconds  # Resetovať začiatočný čas pre 5 minút

        # Každých 10 minút
        if current_time_seconds - global_last_10min_time >= 600:  # Ak ubehlo 10 minút
            max_detections = max(all_detections) if all_detections else 0
            output = f"Dátum a čas spracovania: {current_time}\n"
            output += f"Maximálny počet detekcií osôb za 10 minút: {max_detections}\n"
            write_output(output)
            global_last_10min_time = current_time_seconds  # Resetovať začiatočný čas pre 10 minút

    # Zaznamenanie priemerného a maximálneho počtu detekcií po skončení spracovania každého videa
    if video_detections:
        average_detections = sum(video_detections) / len(video_detections)
        max_detections = max(video_detections)
        output = f"Dátum a čas spracovania: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        output += f"Priemerný počet detekcií osôb v aktuálnom videu: {average_detections:.2f}\n"
        output += f"Maximálny počet detekcií osôb v aktuálnom videu: {max_detections}\n"
        write_output(output)

    # Uvoľnenie video zdroja
    video_capture.release()

# Zaznamenanie celkového priemerného a maximálneho počtu detekcií po skončení spracovania všetkých videí
if all_detections:
    overall_average_detections = sum(all_detections) / len(all_detections)
    overall_max_detections = max(all_detections)
    output = f"Dátum a čas spracovania: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    output += f"Celkový priemerný počet detekcií osôb: {overall_average_detections:.2f}\n"
    output += f"Celkový maximálny počet detekcií osôb: {overall_max_detections}\n"
    write_output(output)

# Zatvorenie všetkých okien
cv2.destroyAllWindows()