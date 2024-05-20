import cv2
import numpy as np
import os
import time
#output do console
# Cesta k priečinku s obrázkami
folder_path = 'images'

# Inicializácia HOG detectora
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

count = 0
start_time = time.time()  # Začiatočný čas

# Prechádzanie všetkých súborov v priečinku
for filename in os.listdir(folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  # Kontrola, či ide o obrázok
        # Načítanie obrázku
        image_path = os.path.join(folder_path, filename)
        image = cv2.imread(image_path)

        # Detekcia osôb v obraze
        boxes, weights = hog.detectMultiScale(image, winStride=(2, 2), padding=(8,8), scale = 1.05)

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
            print(f"Dátum a čas: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Maximálne číslo detekcie osôb za 10 minút: {count}")
            start_time = current_time  # Resetovať začiatočný čas
            count = 0  # Resetovať počet detekcií osôb
        else:
            print(f"Dátum a čas: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(count)
            count = 0  # Resetovať počet detekcií osôb