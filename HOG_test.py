import cv2
import numpy as np

# Načítanie obrázku
image_path = 'images/2f71d24bfd0d30b41e769ef1c0a9635a_L.jpg'
image = cv2.imread(image_path)

# Inicializácia HOG detectora
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detekcia osôb v obraze
boxes, weights = hog.detectMultiScale(image, winStride=(3, 2))
# , padding=(8, 8), scale=1.05

# Práh pre váhy
threshold = 0.3

# Vykreslenie obdĺžnikov okolo detegovaných osôb
for (x, y, w, h), weight in zip(boxes, weights):
    if weight > threshold:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Zobrazenie výsledného obrázka
cv2.imshow('Detekcia osôb pomocou HOG', image)
cv2.waitKey(0)
cv2.destroyAllWindows()