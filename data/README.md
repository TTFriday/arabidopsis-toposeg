# Dataset

The images are not redistributed in this repository.

Download the ChronoRoot 2.0 release from the GigaScience Database:

- DOI: [10.5524/102804](https://doi.org/10.5524/102804)
- Licence: CC0

Use the Arabidopsis seedling segmentation subset and arrange it in nnU-Net format:

```
<data root>/
  imagesTr/
    image_*_0000.png
  labelsTr/
    image_*.png
  imagesTs/
```

Class indices are 0 background, 1 main root, 2 lateral root, 3 seed, 4 hypocotyl, 5 leaf and
6 petiole; index 7 is treated as *ignore*. Only `imagesTr/` and `labelsTr/` enter the frozen
split. `imagesTs/` is used solely by the protocol diagnostics that reproduce the published
assignment.

Point the code at the data root with an absolute path:

```bash
export TOPOSEG_DATA_ROOT=/absolute/path/to/08_image_seedling_segmentation
```

`manifests/splits/main.csv` stores paths relative to that root, so nothing in the release depends
on the directory layout of the machine the experiments were run on.
