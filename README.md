# **Détection d'Émotions avec Furhat**

Ce projet détecte les émotions humaines à partir d'une webcam, affiche une interface graphique, et permet à un robot Furhat de réagir en conséquence.
Les modèles utilisés sont des modèles de DenseNet121 fine tunées
Une API locale Ollama pour générer des réponses empathiques est utilisée
Intègre des gestes animés via l'API Furhat https://docs.furhat.io/remote-api/


## **Scripts Python**

- `camera_emotion_detection_4c.py`  
  → Teste la détection des émotions à travers la caméra via une interface graphique.

- `furhat_emotion_detection_4c.py`  
  → Se connecte au robot via HTTP  
  → Fait réagir Furhat aux émotions détectées (expressions faciales)  
  → Génère des réponses avec l'API Ollama et les fait vocaliser


## **Prérequis**

- Python 3.8+
- PyTorch, OpenCV, PIL, torchvision, requests
- API Furhat (furhat_remote_api)
- Serveur Ollama local actif avec le modèle mistral
- Caméra connectée


## **Dossiers requis**

- `emotion_detection_models/4classes.pth`  
  → Modèle avec 4 classes : `['angry', 'fear', 'happy', 'sad']`

- `emotion_detection_models/5classes.pth`  
  → Modèle avec 5 classes : `['angry', 'disgust', 'fear', 'happy', 'sad']`
