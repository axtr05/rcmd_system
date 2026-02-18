import torch
import pandas as pd
from sklearn import preprocessing
from train import RecSysModel


# =========================
# LOAD DATA + ENCODERS
# =========================

df = pd.read_csv("../input/train_v2.csv")

lbl_user = preprocessing.LabelEncoder()
lbl_movie = preprocessing.LabelEncoder()

df.user = lbl_user.fit_transform(df.user.values)
df.movie = lbl_movie.fit_transform(df.movie.values)

num_users = len(lbl_user.classes_)
num_movies = len(lbl_movie.classes_)


# =========================
# LOAD TRAINED MODEL
# =========================

model = RecSysModel(num_users, num_movies)
model.load_state_dict(torch.load("../recsys_model.pth", map_location="cpu"))
model.eval()


# =========================
# RECOMMEND FUNCTION
# =========================

def recommend(user_id, top_k=5):

    # ---- check user exists ----
    if user_id not in lbl_user.classes_:
        print("User not found in training data.")
        return []

    # ---- encode user id ----
    encoded_user = lbl_user.transform([user_id])[0]

    # ---- movies already watched ----
    rated_movies = df[df.user == encoded_user].movie.values

    all_movies = torch.arange(num_movies)

    # ---- remove watched movies ----
    mask = ~torch.isin(all_movies, torch.tensor(rated_movies))
    movie_tensor = all_movies[mask]

    # ---- create user tensor ----
    user_tensor = torch.full(
        (len(movie_tensor),),
        encoded_user,
        dtype=torch.long
    )

    dummy_rating = torch.zeros(len(movie_tensor))

    # ---- inference ----
    with torch.no_grad():
        scores, _, _ = model(
            user=user_tensor,
            movie=movie_tensor,
            rating=dummy_rating
        )

    scores = scores.squeeze()

    # ---- top K selection ----
    top_scores, top_indices = torch.topk(scores, top_k)

    top_movie_ids = movie_tensor[top_indices].numpy()
    movies = lbl_movie.inverse_transform(top_movie_ids)

    return list(zip(movies, top_scores.numpy()))


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    uid_input = input("Enter user id: ").strip()

    if not uid_input:
        print("Please enter a valid user id.")
        exit()

    uid = int(uid_input)

    recs = recommend(uid, top_k=5)

    print("\nTop recommendations:")
    for movie, score in recs:
        print(f"{movie} → {round(float(score), 2)}")
