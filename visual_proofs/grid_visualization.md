# Map Grid Extraction Visualization

Here is the visualization showing exactly how the map is mathematically divided and extracted by the drone.

![Grid Extraction Visualization](file:///C:/Users/sreev/.gemini/antigravity-ide/brain/e1336a05-f51d-4f27-97f5-bb51e7fb905c/grid_extraction_visualization_1780545944837.png)

### Breakdown:
1. **The Background Image:** That is the original map or satellite photo.
2. **The Faint Global Grid:** That is the map being mathematically downsampled and sliced into the massive `32x32` master grid (or `40x40` depending on your settings).
3. **The Larger Highlighted Square:** That represents the drone's physical location on the map. It cuts out an exact **`11x11`** bubble from the massive grid. This is what the drone actually uses to detect rubble and water.
4. **The Smaller Glowing Center Square:** That is the **`5x5`** thermal sensor grid. It is centered on the exact same spot as the `11x11` grid, but it has a shorter physical range, which is why it sits directly inside the middle of the visual bubble.
