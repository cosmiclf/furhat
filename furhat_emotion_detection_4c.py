from PIL import Image, ImageFont, ImageDraw, Image as PILImage
import cv2
import torch
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import json
import requests
from furhat_remote_api import FurhatRemoteAPI





def draw_interface(frame, label, response, show_response=False):
    h, w = frame.shape[:2]
    sidebar_width = 600

    # Convertir OpenCV (BGR) → PIL (RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.new("RGB", (w + sidebar_width, h), (0, 0, 0))
    pil_image.paste(PILImage.fromarray(frame_rgb), (0, 0))

    draw = ImageDraw.Draw(pil_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 30)  # Tu peux changer de police si besoin
    except:
        font = ImageFont.load_default()

    # Fond sidebar
    draw.rectangle([w, 0, w + sidebar_width, h], fill=(50, 50, 50))
    

    ## Bouton quitter
    quit_btn_top = h - 170
    quit_btn_bottom = h - 110
    draw.rectangle([w + 20, quit_btn_top, w + sidebar_width - 20, quit_btn_bottom], fill=(200, 50, 50))
    draw.text((w + 60, quit_btn_top + 15), "Quitter", font=font, fill=(255, 255, 255))


    # Titre
    draw.text((w + 20, 20), "Détection d'humeur", font=font, fill=(255, 100, 100))

    # Émotion détectée
    if label:
        draw.text((w + 20, 60), f"Émotion : {label}", font=font, fill=(255, 255, 255))

    # Bouton
    send_btn_top = h - 100
    send_btn_bottom = h - 40
    draw.rectangle([w + 20, send_btn_top, w + sidebar_width - 20, send_btn_bottom], fill=(0, 128, 255))
    draw.text((w + 40, send_btn_top + 15), "Envoyer l'humeur", font=font, fill=(0, 0, 0))
    
    # Texte réponse
    # if show_response and response:
    #     lines = []
    #     y_cursor = 120
    #     max_chars = 34
    #     wrapped = [response[i:i+max_chars] for i in range(0, len(response), max_chars)]
    #     for line in wrapped[:10]:
    #         draw.text((w + 20, y_cursor), line, font=font, fill=(255, 255, 255))
    #         y_cursor += 25

    # Convertir retour PIL → OpenCV
    cv2_frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return cv2_frame, (w + 20, send_btn_top, w + sidebar_width - 20, send_btn_bottom, ),  (w + 20, quit_btn_top, w + sidebar_width - 20, quit_btn_bottom)





response = ""            # Pour stocker la réponse de Ollama
show_response = False    # Pour contrôler son affichage dans l'interface
running = True

# === CONFIGURATION ===
MODEL_PATH = "emotion_detection_models/4classes.pth"
CLASSES = ['angry', 'fear', 'happy', 'sad']
NUM_CLASSES = len(CLASSES)
OLLAMA_MODEL = "mistral"
OLLAMA_API = "http://localhost:11434/api/chat"
DISPLAY_ENABLED = True
# === Chargement du modèle ===
print("[INFO] Chargement du modèle...")
try:
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()
    print("[OK] Modèle chargé")
except Exception as e:
    print("[ERREUR] Chargement modèle :", e)
    exit()

# === Prétraitement ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === Connexion Furhat ===
try:
    print("[INFO] Connexion à Furhat...")
    furhat = FurhatRemoteAPI("192.168.10.14")
    furhat.set_led(red=80, green=80, blue=200)
    furhat.set_voice(name='Isabelle-Neural')
    furhat.attend(user="CLOSEST")
    # furhat.say(text="Bonjour, je vais observer ton humeur.")
    print("[OK] Furhat connecté")
except Exception as e:
    print("[ERREUR] Connexion à Furhat :", e)
    exit()


# === Fichiers prompts ===
prompts_files = {
    "prose": "prompts/prompt_3.txt",
    "prompt_0": "prompts/prompt_0.txt",
    "prompt_1": "prompts/prompt_1.txt",
    "prompt_4": "prompts/prompt_4.txt",
    "prompt_emotion": "prompts/prompt_emotion.txt"
}
current_prompt = "prompt_emotion"

# === Gestes robot ===
def angry():
    furhat.set_led(red=255, green=0, blue=0)
    furhat.gesture(body={
        "name": "AngryExpression",
        "class": "furhatos.gestures.Gesture",
        "frames": [
            {
                "time": [0.5],
                "params": {
                    "BROW_DOWN_LEFT": 1.0,
                    "BROW_DOWN_RIGHT": 1.0,
                    "EYE_SQUINT_LEFT": 1.0,
                    "EYE_SQUINT_RIGHT": 1.0,
                    "EXPR_ANGER": 1.0
                }
            },
            {
                "time": [10.0],
                "params": {
                    "reset": True
                }
            }
        ]
    })

def happy():
    furhat.set_led(red=255, green=223, blue=0)
    furhat.gesture(body={
        "name": "HappyExpression",
        "class": "furhatos.gestures.Gesture",
        "frames": [
            {
                "time": [0.5],
                "params": {
                    "BROW_UP_LEFT": 1.0,
                    "BROW_UP_RIGHT": 1.0,
                    "SMILE_OPEN": 0.7,
                    "SMILE_CLOSED": 0.3,
                    "EXPR_ANGER": 0.0
                }
            },
            {
                "time": [10.0],
                "params": {
                    "reset": True
                }
            }
        ]
    })

def sad():
    furhat.set_led(red=70, green=130, blue=180)
    furhat.gesture(body={
        "name": "SadExpression",
        "class": "furhatos.gestures.Gesture",
        "frames": [
            {
                "time": [0.5],
                "params": {
                    "BROW_IN_LEFT": 1.0,
                    "BROW_IN_RIGHT": 1.0,
                    "EXPR_SAD": 1.0,
                    "NECK_TILT": -10.0
                }
            },
            {
                "time": [10.0],
                "params": {
                    "reset": True
                }
            }
        ]
    })

def fear():
    furhat.set_led(red=80, green=26, blue=139)
    furhat.gesture(body={
        "name": "FearExpression",
        "class": "furhatos.gestures.Gesture",
        "frames": [
            {
                "time": [0.5],
                "params": {
                    "EXPR_FEAR": 1.0,
                    "BROW_UP_LEFT": 1.0,
                    "BROW_UP_RIGHT": 1.0,
                    "EYE_SQUINT_LEFT": 0.5,
                    "EYE_SQUINT_RIGHT": 0.5,
                    "NECK_TILT": 5.0,
                    "NECK_PAN": -5.0
                }
            },
            {
                "time": [10.0],
                "params": {
                    "reset": True
                }
            }
        ]
    })

emotion_handlers = {
    'angry': angry,
    'happy': happy,
    'sad': sad,
    'fear': fear
}




def on_mouse(event, x, y, flags, param):
    global response, show_response, running
    if event == cv2.EVENT_LBUTTONDOWN:
        # Bouton envoyer
        x1, y1, x2, y2 = send_button_coords
        if x1 <= x <= x2 and y1 <= y <= y2:
            print(f"[INFO] Émotion envoyée : {label}")
            try:
                furhat.attend(user="CLOSEST")
                if label in emotion_handlers:
                    emotion_handlers[label]()
                response = ask_ollama(label)
                print("[INFO] Réponse générée :", response)
                furhat.say(text=response, blocking=True)
                show_response = True
            except Exception as e:
                print("[ERREUR] Interaction Furhat ou Ollama :", e)

        # Bouton quitter
        qx1, qy1, qx2, qy2 = quit_button_coords
        if qx1 <= x <= qx2 and qy1 <= y <= qy2:
            print("[INFO] Bouton QUITTER cliqué.")
            running = False





# === Détecteur de visages ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Fonction d’appel à Ollama ===
def ask_ollama(emotion):
    print(f"[INFO] Envoi du prompt à Ollama pour émotion : {emotion}")
    system_prompt = f"L'utilisateur homme, semble {emotion}. Réponds-lui avec empathie en français. sans poser de questions à lui et en moins de 3 phrases sois naturel"
    messages = [{"role": "user", "content": system_prompt}]


    output = ""

    try:
        r = requests.post(
            OLLAMA_API,
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
            timeout=15,
            stream=True
        )
        r.raise_for_status()

        for line in r.iter_lines():
            if line:
                try:
                    body = json.loads(line)
                    if "error" in body:
                        print("[ERREUR Ollama] :", body["error"])
                        return "Je suis désolé, je ne peux pas répondre maintenant."
                    if not body.get("done", False):
                        output += body.get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    print("[ERREUR JSON] Ligne mal formée :", line.decode())
                    continue
        return output.strip() if output else "Je ne sais pas quoi dire."

    except Exception as e:
        print("[ERREUR] Appel Ollama échoué :", e)
        return "Je ne parviens pas à répondre pour le moment."




w = 1600
h = 800

# === MAIN ===
cap = cv2.VideoCapture(0)
cv2.namedWindow("Detection d'humeurs", cv2.WINDOW_NORMAL)

if DISPLAY_ENABLED:
    cv2.namedWindow("Detection d'humeurs", cv2.WINDOW_NORMAL)

if not cap.isOpened():
    print("[ERREUR] Impossible d'ouvrir la caméra.")
    exit()

cv2.resizeWindow("Detection d'humeurs", w , h)

print("[INFO] Caméra ouverte. Appuie sur 'Entrée' pour détecter une émotion, ou 'q' pour quitter.")

cv2.setMouseCallback("Detection d'humeurs", on_mouse)


while running:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[ERREUR] Lecture caméra échouée.")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    label = None

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
            print("[ERREUR] Prédiction :", e)
            label = "Erreur"

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        if label:
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)

    # --- Interface graphique ---
    ui_frame, send_button_coords, quit_button_coords = draw_interface(frame, label, response, show_response)

    x1, y1, x2, y2 = send_button_coords

    cv2.imshow("Detection d'humeurs", ui_frame)

    key = cv2.waitKey(20) & 0xFF

    # --- Gestion clavier ---
    if key == ord('q'):
        break

    

    







furhat.say(text="Merci pour ta participation. À bientôt !")
cap.release()
try:
    cv2.destroyAllWindows()
except cv2.error as e:
    print("[ERREUR] Fermeture fenêtres :", e)
