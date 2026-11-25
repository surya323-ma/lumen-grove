# Lumen Grove
*A living particle artwork — real-time computer vision x generative art.*

Step in front of your webcam and become part of a glowing night grove. Your silhouette is rebuilt from thousands of
light particles; your hands are tracked as luminous skeletons that stir the wind, sweep the grass and call fireflies
into orbit. A procedurally grown tree, wind-driven grass, aurora, moon, shooting stars and ground mist complete the scene.

## Run
```
pip install -r requirements.txt
python main.py            # live webcam
python main.py --demo     # no camera: a virtual figure + your mouse
```
Models download automatically on first run (needs internet once).

## Interact
- **Body** – your silhouette becomes particles; embers drift off its edge.
- **Hands** – move them to push grass and branches; fingertips shed sparks; fireflies orbit your palm.
- **Mouse** – works like a hand (great in `--demo`). Click for a burst of light.

## Keys
| Key | Action |
|---|---|
| 1-4 / N | Palette: Electric, Aurora, Ember, Frost (cross-fades) |
| A | Aurora on/off |
| W | Wind: calm / breeze / storm |
| V | Camera preview on/off |
| D | Toggle live camera / demo |
| Space | Pause |
| S | Save a clean PNG to `snapshots/` |
| F | Fullscreen |
| H | Hide / show the interface (installation mode) |
| C | Next camera |
| Q / Esc | Quit |

## Options
`--camera N  --width 840  --art-height 600  --cell 3  --fps 60  --palette 1-4  --seed 7  --no-panel  --no-ui  --fullscreen  --invert  --demo`
`--snapshot out.png:120` renders headless (no window) after 120 frames.

Tip: smaller `--cell` = finer dots but slower. If the mask looks inverted use `--invert`.

## Stack
Python, OpenCV (window + rendering), MediaPipe (segmentation, hand and pose tracking), NumPy.

## Files
`main.py` app loop & controls · `scene.py` world & particles · `camera.py` webcam + vision thread · `hud.py` interface
