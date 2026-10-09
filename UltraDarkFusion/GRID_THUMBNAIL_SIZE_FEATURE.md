# Grid Thumbnail Size Adjustment Feature

## Overview
Added adjustable grid thumbnail sizing to the Dataset Analysis grid view. Users can now resize thumbnails via a slider control in the Display settings, with sizes ranging from 100px to 400px (width).

## Implementation Details

### 1. UI Control (Settings Panel)
- **Location**: Display settings → "Grid thumbnail size:" slider
- **Range**: 100px - 400px (width)
- **Aspect ratio**: Maintained at 2:3 (240x160 default → scales proportionally)
- **Storage key**: `gridThumbnailSize` in QSettings

### 2. How It Works

#### Settings Panel (Lines ~7863-7883)
```python
self.grid_thumbnail_slider = QtWidgets.QSlider(Qt.Horizontal)
self.grid_thumbnail_slider.setRange(100, 400)
self.grid_thumbnail_slider.setValue(max(100, min(400, grid_thumbnail_value)))
self.grid_thumbnail_slider.valueChanged.connect(self.save_grid_thumbnail_size_setting)
```

#### Save Method in SettingsPanel
```python
def save_grid_thumbnail_size_setting(self, value):
    if hasattr(self.parent(), "set_grid_thumbnail_size"):
        self.parent().set_grid_thumbnail_size(value)
    if hasattr(self, "grid_thumbnail_value_label"):
        self.grid_thumbnail_value_label.setText(f"{int(value)} px")
```

#### Update Method in MainWindow (Lines ~44727-44750)
```python
def set_grid_thumbnail_size(self, value, save=True):
    value = int(value)
    value = max(100, min(400, value))
    
    if save and hasattr(self, "settings"):
        self.settings["gridThumbnailSize"] = value
        self.saveSettings()
    
    # Update both model and view with new size
    if hasattr(self, "grid_review_view") and hasattr(self, "grid_thumbnail_model"):
        grid_height = value * 2 // 3
        grid_width = value + 24
        grid_cell_height = grid_height + 50
        
        self.grid_thumbnail_model._thumbnail_size = QtCore.QSize(value, grid_height)
        self.grid_review_view.setIconSize(QtCore.QSize(value, grid_height))
        self.grid_review_view.setGridSize(QtCore.QSize(grid_width, grid_cell_height))
        self.refresh_grid_review_view(preserve_position=True)
```

### 3. Initialization
Grid view is initialized with saved size at startup (Lines ~25912-25930):
```python
grid_thumb_size = int(self.settings.get("gridThumbnailSize", 240))
grid_thumb_size = max(100, min(400, grid_thumb_size))
grid_thumb_height = grid_thumb_size * 2 // 3
grid_cell_width = grid_thumb_size + 24
grid_cell_height = grid_thumb_height + 50

self.grid_review_view.setIconSize(QtCore.QSize(grid_thumb_size, grid_thumb_height))
self.grid_review_view.setGridSize(QtCore.QSize(grid_cell_width, grid_cell_height))
```

### 4. Size Calculation Logic
- **Thumbnail width**: User slider value (100-400px)
- **Thumbnail height**: width × 2/3 (maintains original 240:160 aspect ratio)
- **Grid cell width**: thumbnail width + 24px (for spacing/border)
- **Grid cell height**: thumbnail height + 50px (for label text and spacing)

## Default Behavior
- **Default size**: 240px (original size)
- **Range**: 100px (compact) to 400px (large)
- **Persistence**: Setting is saved and restored on startup

## Features
✓ Smooth slider adjustment
✓ Real-time grid updates
✓ Maintains thumbnail aspect ratio
✓ Preserves scroll position when resizing
✓ Settings persist across sessions
✓ Value label shows current pixel size

## Testing
Use the Grid View in Dataset Analysis to verify:
1. Slider appears in Display settings
2. Dragging slider updates grid size
3. Size persists after closing/reopening
4. Grid scrolling position is maintained
5. All images display at new size
