r"""
test_webrtc.py — Illuminathon: jednoczesny odbiór 3 strumieni WebRTC z telefonów,
segmentacja MediaPipe selfie_segmenter oraz kompozycja barwna RGB:
  - Kamera 1: R (CZERWONY)
  - Kamera 2: G (ZIELONY)
  - Kamera 3: B (NIEBIESKI)
  - Tło: CZARNE

Addytywne mieszanie kolorów (synteza addytywna RGB światła):
  - Czerwony + Zielony    = Żółty (Yellow)
  - Czerwony + Niebieski  = Magenta (Purpura / Fiolet)
  - Zielony + Niebieski   = Cyan (Błękit / Turkus)
  - Czerwony + Zielony + Niebieski = Biały (White)

Sterowanie w oknie kompozycji:
  'q'     - wyjście z programu
  '+'/'-' - regulacja progu pewności detekcji (THRESHOLD)
  'h'     - ukryj / pokaż pasek statusu (przydatne przy rzutniku / pełnym ekranie)
  'f'     - pełny ekran (fullscreen toggle)
"""

import asyncio
import json
import logging
import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from aiohttp import web, WSMsgType
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

PORT = 8080
STATIC_DIR = Path(__file__).parent          # D:\ill\phone.html
FRAME_QUEUE_SIZE = 2                        # Bufor na kamerę: maks 2 klatki
CAMERAS = (1, 2, 3)

CANVAS_WIDTH = 1280
CANVAS_HEIGHT = 720

# Ścieżka do modelu MediaPipe
MODEL_PATH = Path(__file__).with_name("selfie_segmenter.tflite")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Brak pliku modelu: {MODEL_PATH}")

# Ustawienia segmentacji
THRESHOLD = 0.35
MIN_COMPONENT_AREA = 100

SEGMENTER_OPTIONS = mp.tasks.vision.ImageSegmenterOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
    output_confidence_masks=True
)

# Tablica mieszania addytywnego RGB (w formacie BGR dla OpenCV)
# Indeks binarny: bit 0: Cam 1 (Red), bit 1: Cam 2 (Green), bit 2: Cam 3 (Blue)
LUT_RGB = np.array([
    [0,   0,   0],    # 0: Tło czarne
    [0,   0,   255],  # 1: Czerwony (Cam 1 / R)
    [0,   255, 0],    # 2: Zielony (Cam 2 / G)
    [0,   255, 255],  # 3: Żółty (R + G)
    [255, 0,   0],    # 4: Niebieski (Cam 3 / B)
    [255, 0,   255],  # 5: Magenta / Purpura (R + B)
    [255, 255, 0],    # 6: Cyan / Błękit (G + B)
    [255, 255, 255],  # 7: Lśniący Biały (R + G + B)
], dtype=np.uint8)

# Logowanie
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
log = logging.getLogger("illuminathon")

# ---------------------------------------------------------------------------
# Współdzielone zasoby między wątkiem asyncio a wątkiem głównym (OpenCV)
# ---------------------------------------------------------------------------

frame_queues: dict[int, queue.Queue] = {
    cam_id: queue.Queue(maxsize=FRAME_QUEUE_SIZE) for cam_id in CAMERAS
}
latest_frames: dict[int, np.ndarray | None] = {cam_id: None for cam_id in CAMERAS}
camera_status: dict[int, str] = {cam_id: "Oczekiwanie na telefon..." for cam_id in CAMERAS}

stop_event = threading.Event()
active_pcs: dict[int, RTCPeerConnection | None] = {cam_id: None for cam_id in CAMERAS}

# ---------------------------------------------------------------------------
# Konfiguracja WebRTC (STUN)
# ---------------------------------------------------------------------------

RTC_CONFIG = RTCConfiguration(
    iceServers=[
        RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
        RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
    ]
)

# ---------------------------------------------------------------------------
# Czyszczenie i wygładzanie maski (algorytm ze szkielet.py z bezpieczną ramką)
# ---------------------------------------------------------------------------

def clean_mask(confidence: np.ndarray, threshold: float = THRESHOLD, min_area: int = MIN_COMPONENT_AREA) -> np.ndarray:
    """Czyści maskę pewności: progowanie, morfologia, usuwanie szumów i domykanie dziur."""
    mask = (confidence >= threshold).astype(np.uint8) * 255

    # Łączenie bliskich fragmentów i zamykanie małych szczelin
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # Usuwanie małych, odizolowanych plam
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for i in range(1, count):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 255

    # Wypełnianie dziur z bezpieczną ramką 0 (zapobiega zalaniu gdy sylwetka dotyka brzegu)
    flood = cv2.copyMakeBorder(cleaned, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h_f, w_f = flood.shape
    flood_mask = np.zeros((h_f + 2, w_f + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    flood = flood[1:-1, 1:-1]
    enclosed_holes = cv2.bitwise_not(flood)
    cleaned = cv2.bitwise_or(cleaned, enclosed_holes)

    return cleaned

def segment_frame(segmenter, frame: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Wykonuje inferencję MediaPipe na klatce i zwraca binarną maskę sylwetki."""
    if frame is None or frame.size == 0 or frame.shape[0] < 10 or frame.shape[1] < 10:
        return np.zeros((10, 10), dtype=np.uint8)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = segmenter.segment(mp_image)

    if not result.confidence_masks:
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

    confidence = np.squeeze(result.confidence_masks[0].numpy_view())
    if confidence.shape[:2] != frame.shape[:2]:
        confidence = cv2.resize(
            confidence,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

    return clean_mask(confidence, threshold=threshold)

# ---------------------------------------------------------------------------
# Odbiór klatek wideo z toru WebRTC
# ---------------------------------------------------------------------------

async def receive_video(track, cam_id: int, label: str) -> None:
    log.info("Tor wideo aktywny: Kamera %d (%s)", cam_id, label)
    camera_status[cam_id] = "Streaming aktywny"
    q = frame_queues[cam_id]

    while not stop_event.is_set():
        try:
            # Zwiększony timeout (6.0s), aby chwilowe wahania sieci komórkowej nie zrywały wątku
            frame = await asyncio.wait_for(track.recv(), timeout=6.0)
            img = frame.to_ndarray(format="bgr24")

            # Bezpieczne wrzucenie do kolejki
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
            try:
                q.put_nowait(img)
            except queue.Full:
                pass

        except asyncio.TimeoutError:
            log.warning("Timeout: brak klatki przez 6s (Kamera %d, %s) — oczekiwanie...", cam_id, label)
            continue
        except Exception as exc:
            log.info("Tor wideo zakończony (Kamera %d, %s): %s", cam_id, label, exc)
            break

    camera_status[cam_id] = "Rozłączono"
    latest_frames[cam_id] = None

# ---------------------------------------------------------------------------
# Obsługa WebSocket (identyfikacja kamery 1, 2 lub 3)
# ---------------------------------------------------------------------------

async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    cam_param = request.query.get("cam", "1")
    try:
        cam_id = int(cam_param)
        if cam_id not in CAMERAS:
            cam_id = 1
    except ValueError:
        cam_id = 1

    ws = web.WebSocketResponse()
    await ws.prepare(request)
    peer_ip = request.remote
    log.info("Nowe połączenie WebSocket: Kamera %d od %s", cam_id, peer_ip)

    old_pc = active_pcs.get(cam_id)
    if old_pc is not None:
        log.info("Zastępowanie poprzedniego połączenia dla Kamery %d", cam_id)
        try:
            await old_pc.close()
        except Exception:
            pass

    pc = RTCPeerConnection(configuration=RTC_CONFIG)
    active_pcs[cam_id] = pc
    camera_status[cam_id] = "Negocjacja połączenia..."

    @pc.on("track")
    async def on_track(track):
        if track.kind == "video":
            asyncio.ensure_future(receive_video(track, cam_id, peer_ip))

    @pc.on("connectionstatechange")
    async def on_connection_state_change():
        state = pc.connectionState
        log.info("Stan połączenia [Kamera %d / %s]: %s", cam_id, peer_ip, state)
        if state == "connected":
            camera_status[cam_id] = "Streaming aktywny"
        elif state in ("failed", "closed"):
            camera_status[cam_id] = "Błąd / rozłączono"
            try:
                await pc.close()
            except Exception:
                pass
            if active_pcs.get(cam_id) is pc:
                active_pcs[cam_id] = None

    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    log.warning("Nieprawidłowy JSON od Kamery %d (%s)", cam_id, peer_ip)
                    continue

                msg_type = data.get("type")

                if "cam" in data:
                    try:
                        c_num = int(data["cam"])
                        if c_num in CAMERAS:
                            cam_id = c_num
                    except (ValueError, TypeError):
                        pass

                if msg_type == "offer":
                    await pc.setRemoteDescription(
                        RTCSessionDescription(sdp=data["sdp"], type="offer")
                    )

                    answer = await pc.createAnswer()
                    await pc.setLocalDescription(answer)

                    deadline = asyncio.get_event_loop().time() + 10.0
                    while pc.iceGatheringState != "complete":
                        if asyncio.get_event_loop().time() > deadline:
                            log.error("Kamera %d: ICE gathering timeout!", cam_id)
                            break
                        await asyncio.sleep(0.1)

                    await ws.send_json({
                        "type": "answer",
                        "sdp": pc.localDescription.sdp,
                        "cam": cam_id,
                    })
                    log.info("Kamera %d: Odpowiedź SDP wysłana do %s", cam_id, peer_ip)

            elif msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break

    finally:
        log.info("WebSocket zamknięty: Kamera %d (%s)", cam_id, peer_ip)
        try:
            await pc.close()
        except Exception:
            pass
        if active_pcs.get(cam_id) is pc:
            active_pcs[cam_id] = None
            camera_status[cam_id] = "Oczekiwanie na telefon..."

    return ws

# ---------------------------------------------------------------------------
# Serwowanie phone.html
# ---------------------------------------------------------------------------

async def index_handler(request: web.Request) -> web.FileResponse:
    html_path = STATIC_DIR / "phone.html"
    if not html_path.exists():
        raise web.HTTPNotFound(text="Brak pliku phone.html obok test_webrtc.py")
    return web.FileResponse(html_path)

# ---------------------------------------------------------------------------
# Serwer aiohttp
# ---------------------------------------------------------------------------

async def run_server() -> None:
    app = web.Application()
    app.router.add_get("/", index_handler)
    app.router.add_get("/ws", ws_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "0.0.0.0", PORT, reuse_address=True)
        await site.start()
        log.info("Serwer WebRTC gotowy na porcie %d dla 3 kamer.", PORT)
    except OSError as e:
        if getattr(e, 'winerror', None) == 10048 or getattr(e, 'errno', None) == 10048:
            log.error("Port %d jest już zajęty przez inną działającą instancję test_webrtc.py! Zamknij poprzednie okno konsoli.", PORT)
            stop_event.set()
            return
        raise

    while not stop_event.is_set():
        await asyncio.sleep(0.5)

    log.info("Zamykanie aktywnych sesji WebRTC...")
    for cam_id, pc in list(active_pcs.items()):
        if pc is not None:
            try:
                await pc.close()
            except Exception:
                pass
    await runner.cleanup()

def asyncio_thread_fn() -> None:
    asyncio.run(run_server())

# ---------------------------------------------------------------------------
# Mieszanie masek 3 kamer w jedną kompozycję RGB
# ---------------------------------------------------------------------------

def compose_silhouettes(
    masks: dict[int, np.ndarray | None],
    canvas_w: int,
    canvas_h: int,
    lut: np.ndarray
) -> np.ndarray:
    """Skaluje maski do wspólnego płótna i łączy je w jedną kompozycję addytywną RGB."""
    def prepare_mask(m: np.ndarray | None) -> np.ndarray:
        if m is None or m.size == 0:
            return np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        if m.shape[:2] != (canvas_h, canvas_w):
            resized = cv2.resize(m, (canvas_w, canvas_h), interpolation=cv2.INTER_NEAREST)
            return (resized > 0).astype(np.uint8)
        return (m > 0).astype(np.uint8)

    b1 = prepare_mask(masks.get(1))  # R
    b2 = prepare_mask(masks.get(2))  # G
    b3 = prepare_mask(masks.get(3))  # B

    state = b1 | (b2 << 1) | (b3 << 2)
    return lut[state]

# ---------------------------------------------------------------------------
# Pętla wyświetlania: Jedno zunifikowane okno kompozycji RGB
# ---------------------------------------------------------------------------

def display_loop() -> None:
    """Główna pętla inferencji MediaPipe i wyświetlania kompozycji RGB."""
    global THRESHOLD
    window_name = "Illuminathon — Kompozycja RGB (1: Red, 2: Green, 3: Blue)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CANVAS_WIDTH, CANVAS_HEIGHT)

    raw_masks: dict[int, np.ndarray | None] = {cam_id: None for cam_id in CAMERAS}
    show_hud = True
    is_fullscreen = False

    with mp.tasks.vision.ImageSegmenter.create_from_options(SEGMENTER_OPTIONS) as segmenter:
        log.info("MediaPipe ImageSegmenter zainicjalizowany pomyślnie.")

        while not stop_event.is_set():
            try:
                # 1. Sprawdź nowe klatki z kolejek i przelicz maski MediaPipe
                for cam_id in CAMERAS:
                    q = frame_queues[cam_id]
                    try:
                        frame = q.get_nowait()
                        latest_frames[cam_id] = frame
                        try:
                            mask = segment_frame(segmenter, frame, threshold=THRESHOLD)
                            raw_masks[cam_id] = mask
                        except Exception as seg_err:
                            log.warning("Błąd segmentacji dla Kamery %d: %s", cam_id, seg_err)
                    except queue.Empty:
                        pass

                    # Jeśli kamera została trwale rozłączona, wyzeruj maskę
                    if camera_status[cam_id] == "Rozłączono":
                        raw_masks[cam_id] = None

                # 2. Wygeneruj kompozycję barwną addytywnego RGB
                composition = compose_silhouettes(raw_masks, CANVAS_WIDTH, CANVAS_HEIGHT, LUT_RGB)

                # 3. Opcjonalny pasek informacyjny (HUD)
                if show_hud:
                    cv2.putText(composition, f"ILLUMINATHON RGB  |  Prog: {THRESHOLD:.2f} (+/-)  |  'h' ukrywa opis  |  'f' pelny ekran",
                                (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

                    # Statusy kamer: R (Czerwony), G (Zielony), B (Niebieski)
                    status_colors = {
                        1: (0, 0, 255),  # Czerwony BGR
                        2: (0, 255, 0),  # Zielony BGR
                        3: (255, 0, 0),  # Niebieski BGR
                    }
                    cam_names = {1: "Kamera 1 (Red / R)", 2: "Kamera 2 (Green / G)", 3: "Kamera 3 (Blue / B)"}

                    for idx, cid in enumerate(CAMERAS):
                        x_pos = 24 + idx * 400
                        y_pos = CANVAS_HEIGHT - 25
                        st = camera_status[cid]
                        col = status_colors[cid]
                        dot_col = (0, 255, 0) if "Streaming" in st else (80, 80, 80)
                        cv2.circle(composition, (x_pos, y_pos - 5), 6, dot_col, -1)
                        cv2.putText(composition, f"{cam_names[cid]}: {st}",
                                    (x_pos + 15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1, cv2.LINE_AA)

                cv2.imshow(window_name, composition)

                # Bezpieczne sprawdzenie czy okno zostało zamknięte krzyżykiem 'X'
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    log.info("Okno zamknięte przez użytkownika.")
                    break

                # Obsługa klawiszy
                key = cv2.waitKey(15) & 0xFF
                if key == ord("q"):
                    log.info("Wciśnięto 'q' — zamykanie...")
                    break
                elif key in (ord("+"), ord("=")):
                    THRESHOLD = min(0.90, round(THRESHOLD + 0.05, 2))
                    log.info("Próg THRESHOLD: %.2f", THRESHOLD)
                elif key == ord("-"):
                    THRESHOLD = max(0.10, round(THRESHOLD - 0.05, 2))
                    log.info("Próg THRESHOLD: %.2f", THRESHOLD)
                elif key == ord("h"):
                    show_hud = not show_hud
                elif key == ord("f"):
                    is_fullscreen = not is_fullscreen
                    val = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, val)

            except Exception as loop_err:
                log.error("Nieoczekiwany błąd w pętli: %s", loop_err)
                cv2.waitKey(50)

    stop_event.set()
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 68)
    print("  ILLUMINATHON — Zunifikowana Kompozycja RGB")
    print("=" * 68)
    print("Kolory sylwetek:")
    print("  - Kamera 1: R (Czerwony)")
    print("  - Kamera 2: G (Zielony)")
    print("  - Kamera 3: B (Niebieski)")
    print("Synteza addytywna w miejscach nakładania:")
    print("  - R + G = Żółty (Yellow)")
    print("  - R + B = Magenta (Fiolet / Purpura)")
    print("  - G + B = Cyan (Błękit / Turkus)")
    print("  - R + G + B = Biały (White)")
    print()
    print("KLAWISZE STERUJĄCE:")
    print("  'q'     - wyjście z programu")
    print("  '+'/'-' - regulacja progu pewności detekcji (aktualny: %.2f)" % THRESHOLD)
    print("  'h'     - ukryj / pokaż pasek statusu kamer na ekranie")
    print("  'f'     - pełny ekran (fullscreen toggle)")
    print("-" * 68)

    bg_thread = threading.Thread(target=asyncio_thread_fn, daemon=True, name="asyncio-server")
    bg_thread.start()

    try:
        display_loop()
    except KeyboardInterrupt:
        print("\nPrzerwano przez Ctrl+C.")
    finally:
        stop_event.set()
        print("Program zakończony poprawnie.")
