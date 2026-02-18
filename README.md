# DNN Movie Recommendation System

---

## Features

- User & Movie embeddings
- DNN-based rating prediction
- Top-K recommendation system
- Filters already watched movies
- Label encoding for users and movies
- PyTorch training pipeline

---

### imp : this uses Collaborative learning

## Project Structure

```
project/
│
├── input/
│ ├── train_v2.csv_
│ # test and sampleSubmissions csv files can be ignored
│
├── src/
│ ├── train.py # Model training
│ └── rcmd.py # Recommendation script
│
├── recsys_model.pth # Saved model weights
```

## How to Setup in Windows


initially make sure you have python and git in your system
```bash
python --version
git --version

```


Clone Repository

```bash
git clone https://github.com/axtr05/rcmd_system.git
cd rcmd_system
```

1. Create Virtual Environment
```bash
python -m venv venv
```
2. Activate virtual environment:
```bash
venv\Scripts\activate
```

3. Install Dependencies
```bash
pip install torch pandas scikit-learn tez
```

4. go to src dir

```bash
cd src
```

## Train the Model

Run training script:

```bash
python train.py
```

This will generate:
recsys_model.pth

## Get Movie Recommendations

 Run: 
```bash
python recommend.py
```

input any user id that you see in test_v2.csv

#### for example: 

```
for the input of userID : 285

Top recommendations:
2019 → 4.72
318 → 4.71
1148 → 4.68
745 → 4.68
858 → 4.68

the following should be the output 
```






