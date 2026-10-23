r"""
test_blend.py — Illuminathon: Test mieszania addytywnego RGB (bez konieczności używania telefonów).

Trzy sylwetki na czarnym tle:
  - Kamera 1: R (CZERWONA)
  - Kamera 2: G (ZIELONA)
  - Kamera 3: B (NIEBIESKA)

Addytywna synteza światła (RGB):
  - R + G       = ŻÓŁTY (Yellow)
  - R + B       = MAGENTA (Purpura / Fiolet)
  - G + B       = CYAN (Błękit / Turkus)
  - R + G + B   = BIAŁY (White)

Sterowanie:
  Spacja  - Wstrzymaj / wznów płynny ruch sylwetek
  'q'     - Wyjście z programu
"""

import math
import cv2
import numpy as np

# Tablica addytywna RGB w formacie BGR dla OpenCV
LUT_RGB = np.array([
    [0,   0,   0],    # 0: Tło czarne
    [0,   0,   255],  # 1: Czerwony (R / Cam 1)
    [0,   255, 0],    # 2: Zielony (G / Cam 2)
    [0,   255, 255],  # 3: Żółty (R + G)
    [255, 0,   0],    # 4: Niebieski (B / Cam 3)
    [255, 0,   255],  # 5: Magenta (R + B)
    [255, 255, 0],    # 6: Cyan (G + B)
    [255, 255, 255],  # 7: Biały (R + G + B)
], dtype=np.uint8)

def compose_masks(m1: np.ndarray, m2: np.ndarray, m3: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Łączy trzy binarne maski w kompozycję RGB za pomocą szybkiej tablicy LUT."""
    b1 = (m1 > 0).astype(np.uint8)
    b2 = (m2 > 0).astype(np.uint8)
    b3 = (m3 > 0).astype(np.uint8)

    state = b1 | (b2 << 1) | (b3 << 2)
    return lut[state]

def draw_person_silhouette(canvas: np.ndarray, cx: int, cy: int, scale: float = 1.0) -> None:
    """Rysuje syntetyczną sylwetkę człowieka (głowa + ramiona + tułów) w masce."""
    r_head = int(45 * scale)
    cv2.circle(canvas, (cx, cy - int(90 * scale)), r_head, 255, -1)
    axes = (int(80 * scale), int(130 * scale))
    cv2.ellipse(canvas, (cx, cy + int(40 * scale)), axes, 0, 0, 360, 255, -1)

def main():
    width, height = 1000, 700
    window_name = "Illuminathon — Test RGB (Spacja: pauza, q: wyjscie)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)

    animate = True
    t = 0.0

    print("=" * 65)
    print("  ILLUMINATHON — Test mieszania RGB (Addytywny)")
    print("=" * 65)
    print("Kolory kamer:")
    print("  - Kamera 1: R (Czerwony)")
    print("  - Kamera 2: G (Zielony)")
    print("  - Kamera 3: B (Niebieski)")
    print("Nakładanie barw światła:")
    print("  - R + G     -> Żółty")
    print("  - R + B     -> Magenta")
    print("  - G + B     -> Cyan")
    print("  - R + G + B -> Biały")
    print("-" * 65)

    while True:
        m1 = np.zeros((height, width), dtype=np.uint8)
        m2 = np.zeros((height, width), dtype=np.uint8)
        m3 = np.zeros((height, width), dtype=np.uint8)

        center_x = width // 2
        center_y = height // 2 + 30
        radius = 120.0

        angle1 = t * 0.8
        angle2 = t * 0.8 + 2.094  # + 120 deg
        angle3 = t * 0.8 + 4.188  # + 240 deg

        pos1 = (int(center_x + radius * math.cos(angle1)), int(center_y + radius * math.sin(angle1)))
        pos2 = (int(center_x + radius * math.cos(angle2)), int(center_y + radius * math.sin(angle2)))
        pos3 = (int(center_x + radius * math.cos(angle3)), int(center_y + radius * math.sin(angle3)))

        draw_person_silhouette(m1, pos1[0], pos1[1], scale=1.1)
        draw_person_silhouette(m2, pos2[0], pos2[1], scale=1.1)
        draw_person_silhouette(m3, pos3[0], pos3[1], scale=1.1)

        composite = compose_masks(m1, m2, m3, LUT_RGB)

        # Informacje w nagłówku
        cv2.putText(composite, "ILLUMINATHON - TEST ADDYTYWNY RGB", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
        cv2.putText(composite, "Kamera 1: Red  |  Kamera 2: Green  |  Kamera 3: Blue", (30, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 200, 255), 1, cv2.LINE_AA)

        # Legenda barw
        cv2.putText(composite, "K1: Red", (30, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(composite, "K2: Green", (30, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(composite, "K3: Blue", (30, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.putText(composite, "R + G = Zolty", (200, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(composite, "R + B = Magenta", (200, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(composite, "G + B = Cyan", (200, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(composite, "R + G + B = Bialy", (430, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, composite)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            animate = not animate

        if animate:
            t += 0.03

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
