# LiveBot

This is the codes and datasets for the papers: [*LiveBot: Generating Live Video Comments Based on Visual and Textual Contexts*](https://arxiv.org/pdf/1809.04938.pdf).

## What is Live Video Comments?
Live video commenting, which is also called ''video barrage'' (''弹幕'' in Chinese or ''Danmaku'' in Japanese), is an emerging feature on online video sites that allows real-time comments from viewers to fly across the screen like bullets or roll at the right side of the screen.

## Requirements
* Ubuntu 16.0.4
* Python 3.5
* Pytorch 0.4.1
* Sklearn >= 0.19.1

## Datasets

- **Processed dataset** can be directly used for our codes to reproduce the results reported in the paper. It should be downloaded from [Google Drive](https://drive.google.com/open?id=13hLJ4yCJVJjz02YB0dyugeE_fcl6EI-B) or [Baidu Pan](https://pan.baidu.com/s/1xdfnZKtBpESEuLvhyU0RBw), and put in the folder */data*.

- **Raw dataset** consists of the videos and the corresponding live comments that directly downloaded from the Bilibili video websites. It can be found at [Google Drive](https://drive.google.com/open?id=15m5SbD-2ByaAr9Ik_vhL2GuUseVR-_EB) or [Baidu Pan](https://pan.baidu.com/s/1WSDbopxTMoxOAsd29gT77A). After processed with the scripts in the folder */data*, it can be transformed into the processed datasets above.

## Livebot Model

- Step 1: Download the processed dataset above
- Step 2: Train a model 
    ```
    python3 codes/transformer.py -mode train -dir CKPT_DIR
    ```

- Step 3: Restore the checkpoint and evaluate the model
    ```
    python3 codes/transformer.py -mode test -restore CKPT_DIR/checkpoint.pt -dir CKPT_DIR
    ```

## Process a raw dataset (Optional)

- Step 1: Extract the frames from the videos and the comments from the .ass files.
    ```
    python3 data/extract.py
    ```
- Step 2: Convert the extracted images and text into the format required by our model.
    ```
    python3 data/preprocess.py
    ```
- Step 3: Construct the candidate set for the evaluation of the model.
    ```
    python3 data/add_candidate.py
    ```

## Note

- More details regarding the model and the dataset can be found in our paper.

- The code is currently non-deterministic due to various GPU ops, so you are likely to end up with a slightly better or worse evaluation.

## Citation

Hopefully the codes and the datasets are useful for the future research. If you use the above codes or datasets for your research, please kindly cite our paper:
```
@inproceedings{livebot,
  author    = {Shuming Ma and
               Lei Cui and
               Damai Dai and
               Furu Wei and
               Xu Sun},
  title     = {LiveBot: Generating Live Video Comments Based on Visual and Textual Contexts},
  booktitle = {{AAAI} 2019},
  year      = {2019}
}
```