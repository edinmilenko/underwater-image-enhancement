import cv2
import os

def video_divider(file_name):
    # Percorso del video da elaborare

    # Cartella di output per i frame
    output_folder = 'imgae'
    os.makedirs(output_folder, exist_ok=True)

    # Carica il video
    cap = cv2.VideoCapture(file_name)

    # Controllo apertura
    if not cap.isOpened():
        print("Errore: impossibile aprire il video.")
        exit()

    # Conta i frame
    frame_count = 0

    while True:
        # Legge un frame
        ret, frame = cap.read()
        if not ret:
            break  # fine del video

        # Nome file (es: frame_0001.jpg)
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")

        if frame_count % 20 == 0:
            cv2.imwrite(frame_filename, frame)
            # (opzionale) mostra avanzamento
            print(f"Salvato: {frame_filename}")

        frame_count += 1
    # Rilascia risorse
    cap.release()
    print(f"✅ {frame_count} frame salvati in '{output_folder}'")