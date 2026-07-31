from sklearn.model_selection import train_test_split

def split_data(df, target, test_size=0.3, seed=42):
    """
    Split the dataframe into train and test sets.
    """
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed)
