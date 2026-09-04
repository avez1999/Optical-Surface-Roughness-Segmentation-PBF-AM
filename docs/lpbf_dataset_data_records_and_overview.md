# Data Records and Data Overview

## Data Records

The dataset and reference implementation are released together in the accompanying GitHub repository. The repository preserves a one-to-one correspondence between each optical PNG micrograph and its JSON polygon annotation and contains the files required to reproduce the reference semantic-segmentation experiments. The tagged GitHub release used for the reference experiments should be cited in the article.

The release is organized to preserve a one-to-one correspondence between each optical PNG micrograph and its JSON polygon annotation. It contains separate CDF-processed (hereafter, polished) and as-built subsets. Within each subset, the PNGs directory contains native-resolution optical surface images and the JSONs directory contains the paired polygon annotations. Each JSON record stores the feature class and ordered polygon vertices for every annotated region associated with an image.

The public repository contains 61 paired optical-image and annotation files: 48 polished images and 13 as-built images. It also includes two per-image CSV manifests, one for each subset. The manifests document the fixed reference partition created with seed 42 and a 70/20/10 training-validation-test split: 33/9/6 images for polished data and 9/2/2 images for as-built data. The reference training code reproduces the split from sorted image-annotation pairs and the fixed seed; the manifests provide a human-readable record of the assignment.

The release further includes a machine-readable dataset-statistics report, class maps, the two selected best-validation-loss reference U-Net checkpoints, and their associated epoch-wise training histories. Held-out inference exports are provided for eight test images, comprising six polished and two as-built examples. Each export set contains the optical image, the rendered manual-annotation overlay, and the rendered model-prediction overlay for qualitative inspection. The annotations are used only after inference to quantify agreement with the reference masks; they are not supplied to the model as an input.

Table 2. Files and resources included in the public GitHub repository.

Optical micrographs | PNG | 61 | Native-resolution optical surface images: 48 polished and 13 as-built.

Polygon annotations | JSON | 61 paired files | Feature labels and ordered polygon x,y coordinates paired with the optical images.

Image manifests | CSV | 2 | Per-image dataset membership and fixed training, validation, or test assignment.

Dataset statistics | JSON | 1 | Reproducible image-level and rasterized class-coverage statistics.

Reference U-Net checkpoints | PT | 2 | Selected best-validation-loss checkpoints for polished and as-built segmentation.

Training histories | JSON and CSV | 2 pairs | Epoch-wise training and validation loss and mean-IoU records for the reference retraining runs.

Held-out inference exports | PNG overlays | 8 image sets | Optical, manual-annotation overlay, and model-prediction overlay for each held-out test image.

Training and evaluation code | Python | Repository | Annotation rasterization, partitioning, model training, inference, evaluation, and statistics generation.

## Data Overview

The release contains 61 LPBF optical surface images spanning four surface-view categories: dome, slope, side, and top. The polished subset comprises 48 images distributed evenly across the four view categories. The as-built subset comprises 13 images, including four dome, four slope, four side, and one top view. The polished images span four coupon groups and twelve named processing-condition groups; the condition codes are retained exactly as provided in the source filenames.

The polished subset contains three foreground classes: Contour, Polished, and Powder. The as-built subset contains Anomaly, Contour, and Powder. Across both subsets, the release contains 37,337 polygon-level feature annotations. Rasterization of the polygons demonstrates substantial class imbalance. Powder occupies 0.30% of pixels in the polished subset, while Anomaly occupies 0.38% of pixels in the as-built subset. These rare classes should therefore be evaluated with class-specific overlap and detection metrics rather than pixel accuracy alone.

Table 3. Rasterized class-pixel coverage across the complete dataset before partitioning.

Polished | Background: 37,153,562 (31.0%) | Contour: 32,263,373 (26.9%) | Polished: 50,156,183 (41.8%) | Powder: 363,906 (0.30%) | Anomaly: not applicable.

As-built | Background: 14,883,019 (46.4%) | Contour: 16,244,019 (50.6%) | Polished: not applicable | Powder: 851,300 (2.65%) | Anomaly: 120,990 (0.38%).

Figure 3. Dataset organization and representative class coverage. The figure should show the repository-level folder structure and representative optical images with the polished and as-built label schemes. Detailed numerical class-coverage statistics are reported in Table 3 and in the machine-readable statistics report included in the repository.
