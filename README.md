# Math Mnist

## Requirements

Recommend using python3 and a virtual env.
```
conda create -n myenv python=3.6
source activate myenv
pip install -r requirements.txt
```

## Task

Given  [kaggle](https://www.kaggle.com/xainano/handwrittenmathsymbols) handwritten math symbols dataset prepare data pipeline for future analysis.


## Download the dataset

We will used the handwritten math symbols dataset from kaggle. Download it [here](https://www.kaggle.com/xainano/handwrittenmathsymbols).

Here is the raw structure of the data:
```
extracted_images/
    log/
        filename.jpg
        ...
    sigma/
        filename.jpg
        ...
```

The dataset contains 375974 images belonging to 82 labels.

Once the download is complete, move the dataset into MathMnist/data:
```bash
mv path/to/extracted_images/ path/to/MathMnist/data/
```

Run the script `build_dataset.py` which will resize the images to size `(28, 28)`. The new resized dataset will be located by default in `data_split`:

```bash
python build_dataset.py
```

## Todo

- `build_dataset.py` seems to be so slow, make code more efficient
- probably bug in test/val split


