# Underwater Image Enhancement

Underwater image enhancement with WaterNet, sharpening and CLAHE.

```bash
pip install -r requirements.txt
python main.py
```

Place images in `data/input/left/` and `data/input/right/`. Results are saved in the matching folders under `data/output/`.

Code: `underwater_enhancement/` · Weights: `models/weights.pt` · Selected examples: `examples/`.

| Input | Output |
| :---: | :---: |
| ![Vegetation: input](examples/input/left_1771928675.jpg) | ![Vegetation: output](examples/output/left_1771928675.jpg) |
| ![Algae: input](examples/input/left_1771928939.jpg) | ![Algae: output](examples/output/left_1771928939.jpg) |
| ![Rocks: input](examples/input/right_1771928923.jpg) | ![Rocks: output](examples/output/right_1771928923.jpg) |
