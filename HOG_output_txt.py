import cv2
import numpy as np
import os
import time
import datetime

# Cesta k priečinku s obrázkami
folder_path = 'tt'

# Inicializácia HOG detectora
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

start_time = time.time()  # Začiatočný čas

# Funkcia na zápis výstupu do textového súboru
def write_output(output):
    with open("20240514_output.txt", "a") as file:
        file.write(output + "\n")




# Prechádzanie všetkých súborov v priečinku
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Kontrola, či ide o obrázok
        # Získanie dátumu a času vytvorenia súboru
        image_path = os.path.join(folder_path, filename)
        creation_time = os.path.getctime(image_path)
        creation_time_formatted = datetime.datetime.fromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S')

        # Načítanie obrázku
        image = cv2.imread(image_path)

        # Detekcia osôb v obraze
        boxes, weights = hog.detectMultiScale(image, winStride=(1,1), padding=(8,8), scale = 1.05)

        # Vykreslenie obdĺžnikov okolo detegovaných osôb
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count += 1

        # Zobrazenie výsledného obrázka
        cv2.imshow('Detekcia osôb pomocou HOG', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        current_time = time.time()  # Aktuálny čas
        if current_time - start_time >= 600:  # Ak ubehlo 10 minút
            output = f"Dátum a čas vytvorenia fotografie: {creation_time_formatted}\n"
            output += f"Dátum a čas spracovania: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            output += f"Maximálne číslo detekcie osôb za 10 minút: {str(count)}\n"  # Prevod na reťazec
            write_output(output)
            start_time = current_time  # Resetovať začiatočný čas
            count = 0  # Resetovať počet detekcií osôb
        else:
            output = f"Dátum a čas vytvorenia fotografie: {creation_time_formatted}\n"
            output += f"Dátum a čas spracovania: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            output += f"Počet detekcií osôb: {str(count)}\n"  # Prevod na reťazec
            write_output(output)
            count = 0  # Resetovať počet detekcií osôb
