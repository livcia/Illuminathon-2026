# Illuminathon — Interaktywna Instalacja RGB (WebRTC + MediaPipe)

Projekt artystyczno-technologiczny: pobiera obraz na żywo z 3 telefonów przez internet (WebRTC), segmentuje sylwetki osób (MediaPipe Tasks) i nakłada je na czarnym tle w modelu addytywnym RGB:
- **Kamera 1:** R (Czerwony)
- **Kamera 2:** G (Zielony)
- **Kamera 3:** B (Niebieski)
- **Mieszanie w nakładaniu:** R+G = Żółty, R+B = Magenta, G+B = Cyan, R+G+B = Biały.

---

## 🚀 Szybki start na hackathonie (nowy komputer)

### 1. Pobierz repozytorium
```powershell
git clone <URL_REPO>
cd ill
```

### 2. Zainstaluj biblioteki
```powershell
python -m pip install -r requirements.txt
```

*(Uwaga: upewnij się, że nie masz jednocześnie zwykłego `opencv-python` i `opencv-contrib-python`. W `requirements.txt` jest tylko wersja `contrib`).*

---

## 🎮 Uruchamianie (2 Terminale)

### Terminal 1: Aplikacja główna (Python)
```powershell
python test_webrtc.py
```
*Otworzy się okno kompozycji z czarnym tłem i statusem oczekiwania na 3 telefony.*

### Terminal 2: Tunel Cloudflare (dla telefonów przez internet)
```powershell
.\start_tunnel.ps1
```
*(Skrypt sam pobierze `cloudflared.exe` jeśli go nie ma i uruchomi tunel).*

---

## 📱 Podłączanie telefonów

Skopiuj wygenerowany link HTTPS (np. `https://losowe-slowa.trycloudflare.com`):

* **Telefon 1:** `https://losowe-slowa.trycloudflare.com/?cam=1`  *(Czerwona sylwetka)*
* **Telefon 2:** `https://losowe-slowa.trycloudflare.com/?cam=2`  *(Zielona sylwetka)*
* **Telefon 3:** `https://losowe-slowa.trycloudflare.com/?cam=3`  *(Niebieska sylwetka)*

Kliknij **▶ Start kamery** i zezwól na dostęp. Obraz pojawi się na ekranie komputera.

---

## ⌨️ Klawisze w oknie kompozycji

| Klawisz | Akcja |
|---------|-------|
| `f` | Pełny ekran (fullscreen toggle — idealne na rzutnik / scenę) |
| `h` | Ukryj / pokaż pasek statusu i opis |
| `+` / `-` | Regulacja czułości wycinania tła (próg `THRESHOLD`) |
| `q` | Zamknięcie programu i rozłączenie kamer |

---

## 🧪 Szybki test offline (bez sieci i telefonów)

Aby sprawdzić działanie mieszania kolorów i wyświetlania bez podłączania telefonów:
```powershell
python test_blend.py
```

---

## ⚠️ Najczęstsze problemy na hackathonie

1. **`WinError 10048` (adres zajęty):**
   Uruchomiłeś `python test_webrtc.py` w dwóch oknach naraz. Zamknij poprzednie okno terminala.

2. **`Bad Gateway (502)` / Odmowa połączenia:**
   Telefon próbuje wejść na stronę, gdy `test_webrtc.py` nie jest włączony. **Zawsze najpierw włącz Pythona w Terminalu 1, a dopiero potem wchodź na link z telefonu.**

3. **Przeglądarka na telefonie:**
   Link musi zaczynać się od **`https://`**, w przeciwnym razie mobilne przeglądarki (Safari, Chrome) blokują dostęp do kamery ze względów bezpieczeństwa.
