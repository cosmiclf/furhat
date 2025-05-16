import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

# === CONFIGURATION ===
MODEL_PATH = "emotion_detection_models/4classes.pth"
CLASSES = ['angry', 'fear', 'happy', 'sad']
NUM_CLASSES = len(CLASSES)

# === Chargement du modèle fine-tuné ===
model = models.densenet121()
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

# === Prétraitement pour ResNet ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Détecteur de visages ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Webcam ===
cap = cv2.VideoCapture(0)
cv2.namedWindow("Détection d'humeurs", cv2.WINDOW_NORMAL)  # assure qu'une seule fenêtre est utilisée

if not cap.isOpened():
    print("Erreur : impossible d'ouvrir la webcam.")
    exit()

print("Appuie sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Erreur de capture de la webcam.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]

        try:
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            input_tensor = transform(face_pil).unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                label = CLASSES[predicted.item()]
        except Exception as e:
            print(f"Erreur dans la prédiction : {e}")
            label = "Erreur"

        # Affichage du rectangle et du label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    cv2.imshow("Détection d'humeurs", frame)

    # Attend 20 ms pour réduire la charge système et intercepter la touche 'q'
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()

try:
    cv2.destroyAllWindows()
except cv2.error as e:
    print("Erreur lors de la fermeture des fenêtres :", e)
