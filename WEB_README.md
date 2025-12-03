# Værktøjsdetektion Web Interface

Dette system bruger YOLOv8 til at detektere værktøj i realtid via webcam og vise resultaterne på en hjemmeside.

## Værktøjsklasser
- **0** = Hammer (skal placeres i øverste venstre hjørne)
- **1** = Screwdriver (skal placeres i øverste højre hjørne)
- **2** = Wrench (skal placeres i nederste venstre hjørne)

## Installation

1. Installer de nødvendige pakker (hvis ikke allerede gjort):
```bash
pip install flask flask-cors ultralytics opencv-python
```

## Sådan køres systemet

1. **Start web serveren:**
```bash
python detect_camera_web.py
```

2. **Åbn browseren:**
   - Gå til: `http://localhost:5000`
   - Du vil se live kamera feed og detektionsresultater

## Hvad sker der?

1. Programmet åbner dit webcam
2. YOLO modellen analyserer billedet i realtid
3. Når et værktøj detekteres (hammer, screwdriver eller wrench):
   - Det vises på hjemmesiden med navn og confidence score
   - Canvas'en viser automatisk hvor værktøjet skal placeres (grøn firkant)
4. Du kan også indtaste værktøjsnummeret manuelt (0, 1 eller 2)

## Funktioner på hjemmesiden

- **Live kamera feed**: Viser realtid video med detektioner markeret
- **Registreret værktøj**: Viser det detekterede værktøj og confidence score
- **Placeringsguide**: Canvas med 4 positioner der viser hvor værktøjet skal placeres
- **Manuel indtastning**: Du kan også teste systemet ved at indtaste 0, 1 eller 2 manuelt

## Tekniske detaljer

- Flask web server kører på port 5000
- Detektioner opdateres hver 500ms
- Video stream kører med ~30 FPS
- Systemet viser kun den mest sikre detektion af gangen

## Fejlfinding

**Problem**: "Model not found"
- **Løsning**: Træn modellen først med `python train.py`

**Problem**: "Could not open camera"
- **Løsning**: Tjek at dit webcam er tilsluttet og ikke bruges af et andet program

**Problem**: Hjemmesiden viser ikke video
- **Løsning**: Tjek at detect_camera_web.py kører og at du kan se "Camera detection started!" i terminalen

## Arkitektur

```
detect_camera_web.py    → Python Flask server + YOLO detection
├── /video_feed         → Video stream endpoint
├── /detection          → JSON API for seneste detektion  
└── /                   → Hovedside (index.html)

web/index.html          → Frontend interface
├── Live video display
├── Detection info
└── Canvas placement guide
```
