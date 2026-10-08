USER = "userId"
MOVIE = "movieId"
RATING = "rating"
TIMESTAMP = "timestamp"
TITLE = "title"
GENRES = "genres"
TAG = "tag"
LABEL = "label"
SCORE = "score"
SPLIT = "split"
CAND_ID = "cand_id"
TARGET_TS = "target_ts"

RATINGS_COLUMNS = [USER, MOVIE, RATING, TIMESTAMP]
MOVIES_COLUMNS = [MOVIE, TITLE, GENRES]
TAGS_COLUMNS = [USER, MOVIE, TAG, TIMESTAMP]
LINKS_COLUMNS = [MOVIE, "imdbId", "tmdbId"]

TRAIN = "train"
VAL = "val"
TEST = "test"
SPLITS = [TRAIN, VAL, TEST]

RATINGS_DTYPES = {
    USER: "int64",
    MOVIE: "int64",
    RATING: "float32",
    TIMESTAMP: "int64",
}
