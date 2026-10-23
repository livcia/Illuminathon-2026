
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("selfie_segmenter.tflite")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Brak modelu: {MODEL_PATH}")

# Ustawienia do strojenia
THRESHOLD = 0.35
MIN_COMPONENT_AREA = 100

options = mp.tasks.vision.ImageSegmenterOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    output_confidence_masks=True
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    raise RuntimeError("Nie można otworzyć kamery.")

def clean_mask(confidence):
    # Niższy próg może odzyskać słabiej wykryte fragmenty.
    mask = (confidence >= THRESHOLD).astype(np.uint8) * 255

    # Łączenie bliskich fragmentów i zamykanie małych szczelin.
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (7, 7)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel_close
    )

    # Usuwanie małych, odizolowanych plam.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    cleaned = np.zeros_like(mask)

    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= MIN_COMPONENT_AREA:
            cleaned[labels == i] = 255

    # Wypełnianie dziur całkowicie otoczonych sylwetką.
    flood = cleaned.copy()
    h, w = flood.shape
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    enclosed_holes = cv2.bitwise_not(flood)
    cleaned = cv2.bitwise_or(cleaned, enclosed_holes)

    return cleaned

try:
    with mp.tasks.vision.ImageSegmenter.create_from_options(
        options
    ) as segmenter:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = segmenter.segment(mp_image)

            if not result.confidence_masks:
                continue

            confidence = result.confidence_masks[0].numpy_view()


            confidence = cv2.resize(
                confidence,
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

            # TEST: porównanie trzech progów
            for threshold in (0.25, 0.35, 0.45):
                test_mask = (
                    confidence >= threshold
                ).astype(np.uint8) * 255

                cv2.imshow(
                    f"Prog {threshold:.2f}",
                    test_mask
                )

            mask = clean_mask(confidence)


            # Diagnostyka: porównaj maskę i obraz końcowy.
            output = np.zeros_like(frame)
            output[mask > 0] = (128, 128, 128)

            cv2.imshow("1 - Kamera", frame)
            cv2.imshow("2 - Maska sylwetki", mask)
            cv2.imshow("3 - Szara sylwetka", output)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            # Regulacja progu bez restartu:
            # - zmniejsz próg: więcej pikseli zostanie wykrytych
            # - zwiększ próg: mniej pikseli, zwykle mniej tła
            elif key == ord("-"):
                THRESHOLD = max(0.10, THRESHOLD - 0.05)
            elif key == ord("+") or key == ord("="):
                THRESHOLD = min(0.90, THRESHOLD + 0.05)

finally:
    cap.release()
    cv2.destroyAllWindows()
