# Data Dictionary

## Image and annotation pairing

Each optical micrograph in a subset's `PNGs/` directory is paired with a JSON file of the same stem in `JSONs/`. For example, `Coupon10-dome-30min150.png` is paired with `Coupon10-dome-30min150.json`.

## JSON annotation schema

Each JSON file is a VIA-style mapping whose top-level key is the image filename. The associated record contains the source filename, file size in bytes, and `regions`.

Each region has the following fields:

| Field | Meaning |
| --- | --- |
| `shape_attributes.name` | Always `polygon` for the supplied feature regions. |
| `shape_attributes.all_points_x` | Ordered horizontal polygon-vertex coordinates in image pixels. |
| `shape_attributes.all_points_y` | Ordered vertical polygon-vertex coordinates in image pixels. |
| `region_attributes.label` | Semantic class assigned to the polygon. |

The `all_points_x` and `all_points_y` arrays have the same length and are interpreted pairwise in native image coordinates, with the image origin at the upper left.

## Class maps

| Subset | Class index to label |
| --- | --- |
| Polished | `0: background`, `1: Contour`, `2: Polished`, `3: Powder` |
| As-built | `0: background`, `1: Anomaly`, `2: Contour`, `3: Powder` |

Polygons are rasterized in the order recorded in the JSON. Where polygons overlap, a later polygon can overwrite an earlier class in the final semantic mask. The training code implements this rasterization in `unet/train_unet.py`.
