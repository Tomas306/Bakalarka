import cv2
import numpy as np
import os
import time
import datetime

# Cesta k priečinku s obrázkami
folder_path = ('tt')

# Inicializácia HOG detectora
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Načítanie masky
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)  # Predpokladáme, že maska je čiernobiela

start_time = time.time()  # Začiatočný čas
last_5min_time = start_time  # Začiatočný čas pre 5 minútový interval
last_10min_time = start_time  # Začiatočný čas pre 10 minútový interval

detections_5min = []  # Zoznam na uchovanie počtu detekcií za 5 minút
detections_10min = []  # Zoznam na uchovanie počtu detekcií za 10 minút

# Funkcia na zápis výstupu do textového súboru
def write_output(output):
    with open("20240514_output.txt", "a") as file:
        file.write(output + "\n")

while True:
    count = 0  # Resetovať počet detekcií osôb

    # Prechádzanie všetkých súborov v priečinku
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Kontrola, či ide o obrázok
            # Získanie dátumu a času vytvorenia súboru
            image_path = os.path.join(folder_path, filename)
            creation_time = os.path.getctime(image_path)
            creation_time_formatted = datetime.datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')

            # Načítanie obrázku
            image = cv2.imread(image_path)

            # Deaktivácia detekcie v určených oblastiach pomocou masky
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            # Detekcia osôb v obraze
            boxes, weights = hog.detectMultiScale(masked_image, winStride=(2,2), padding=(8,8))

            # Vykreslenie obdĺžnikov okolo detegovaných osôb
            for (x, y, w, h) in boxes:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                count += 1


            # Zobrazenie výsledného obrázka
            cv2.imshow('Detekcia osôb pomocou HOG', image)
            # cv2.waitKey(1000)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            current_time = time.time()  # Aktuálny čas
            current_time_seconds = time.time() # Aktuálny čas

            output = f"Dátum a čas vytvorenia fotografie: {creation_time_formatted}\n"
            output += f"Dátum a čas spracovania: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            output += f"Počet detekcií osôb: {count}\n"
            write_output(output)

            detections_5min.append(count)
            detections_10min.append(count)

            # Každých 5 minút
            if current_time_seconds - last_5min_time >= 300:  # Ak ubehlo 5 minút
                average_detections_5min = sum(detections_5min) / len(detections_5min) if detections_5min else 0
                output = f"Dátum a čas spracovania: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                output += f"Priemerný počet detekcií osôb za 5 minút: {average_detections_5min:.2f}\n"
                write_output(output)
                last_5min_time = current_time_seconds  # Resetovať začiatočný čas pre 5 minút
                detections_5min = []  # Resetovať zoznam detekcií za 5 minút

            # Každých 10 minút
            if current_time_seconds - last_10min_time >= 600:  # Ak ubehlo 10 minút
                max_detections_10min = max(detections_10min) if detections_10min else 0
                output = f"Dátum a čas spracovania: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                output += f"Maximálny počet detekcií osôb za 10 minút: {max_detections_10min}\n"
                write_output(output)
                last_10min_time = current_time_seconds  # Resetovať začiatočný čas pre 10 minút
                detections_10min = []  # Resetovať zoznam detekcií za 10 minút

    # time.sleep(5)