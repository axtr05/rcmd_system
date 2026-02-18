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

Clone Repository

```bash
git clone https://github.com/yourusername/rcmd_system.git
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

go to src dir

```bash
cd src
```

## Train the Model

Run training script:

```bash
python src/train.py
```

This will generate:
recsys_model.pth

## Get Movie Recommendations

Run: 
```bash
python src/rcmd.py
```

