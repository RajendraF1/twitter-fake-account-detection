import re
import pandas as pd


URL_PATTERN = re.compile(r"https?://[^\s]+|www\.[^\s]+")


def add_follower_friend_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    friends = df["friends_count"].fillna(0).astype(int) + 1
    df["follower_friend_ratio"] = df["followers_count"].fillna(0).astype(int) / friends
    return df


def add_screen_name_numeric_ratio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "screen_name" in df.columns:
        screen_name = df["screen_name"].fillna("").astype(str)
        total_chars = screen_name.apply(len).replace(0, 1)
        numeric_chars = screen_name.apply(lambda x: sum(c.isdigit() for c in x))
        df["screen_name_numeric_ratio"] = numeric_chars / total_chars
    else:
        df["screen_name_numeric_ratio"] = 0
    return df


def add_url_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    description = df["description"].fillna("").astype(str)
    has_url = description.apply(lambda x: 1 if URL_PATTERN.search(x) else 0)
    df["url_available"] = has_url
    df["description_has_url"] = has_url
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_follower_friend_ratio(df)
    df = add_screen_name_numeric_ratio(df)
    df = add_url_features(df)
    return df


NEW_FEATURE_COLUMNS = [
    "follower_friend_ratio",
    "screen_name_numeric_ratio",
    "url_available",
    "description_has_url",
]
