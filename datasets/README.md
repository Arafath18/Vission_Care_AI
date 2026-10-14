# Kermany OCT Dataset (included)

VisionCare uses a retinal OCT dataset layout organized by class folders.

## Layout

```
datasets/kermany/
  train/{CNV,DME,DRUSEN,NORMAL,AMD,CSCR,ERM,MH,RD,GLAUCOMA,RP}/
  test/{CNV,DME,DRUSEN,NORMAL,AMD,CSCR,ERM,MH,RD,GLAUCOMA,RP}/
```

## Notes

- The project is configured for 11 retinal disease labels.
- The included `scripts/init_dataset_structure.py` script will create the empty class folders for all configured labels.
- The Kermany OCT dataset download scripts currently provide the original 4-class OCT dataset (`CNV`, `DME`, `DRUSEN`, `NORMAL`).
- For the additional labels (`AMD`, `CSCR`, `ERM`, `MH`, `RD`, `GLAUCOMA`, `RP`), add labeled images manually or source a compatible multi-class OCT dataset.

## API

`GET /api/dataset/info` returns class counts and paths when the server is running.

## Re-download / initialize

```bash
python scripts/init_dataset_structure.py
python scripts/download_datasets.py --source hf-stream --max-per-class 500
python scripts/download_datasets.py --source hf-parquet --max-per-class 500
python scripts/download_datasets.py --source kaggle --kaggle-dataset andrewmvd/retinal-oct-categorization --max-per-class 500
python scripts/prepare_kaggle_split.py --train-per-class 800 --test-per-class 250
```

If you want to use Kaggle as the source, add a Kaggle token at `%USERPROFILE%\.kaggle\kaggle.json` or set `KAGGLE_USERNAME` and `KAGGLE_KEY` in your environment.

If you have an external dataset for the missing disease classes, use:

```bash
python scripts/add_external_dataset.py --source-dir /path/to/external/dataset
```

If the external dataset contains top-level `train/` and `test/` folders, import them directly:

```bash
python scripts/add_external_dataset.py --source-dir /path/to/external/dataset --split train-test
```

If the external dataset contains only class folders, split them into `train/` and `test/` automatically:

```bash
python scripts/add_external_dataset.py --source-dir /path/to/external/dataset --split auto --train-ratio 0.8
```

The external source should contain class folders named for the target labels, for example `AMD`, `CSCR`, `ERM`, `MH`, `RD`, `GLAUCOMA`, `RP`. Existing dataset images are preserved.
