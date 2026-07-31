

def prepare_vand(raw):
    """
    Cleans the raw Vanderbilt dataset:
    - Derives BMI and Waist-Hip ratio
    - Maps gender to binary and diabetes target based on glyhb >= 7.0
    - Keeps relevant features, drops missing values, and resets index.
    """
    d = raw.copy()
    d['BMI']        = 703 * d['weight'] / (d['height'] ** 2)
    d['waist_hip']  = d['waist'] / d['hip']
    d['gender_bin'] = (d['gender'].str.strip().str.lower() == 'male').astype(int)
    d['Diabetes']   = (d['glyhb'] >= 7.0).astype(int)
    keep = ['chol', 'stab.glu', 'hdl', 'ratio',
            'age', 'gender_bin', 'height', 'weight',
            'BMI', 'bp.1s', 'bp.1d', 'waist', 'hip', 'waist_hip', 'Diabetes']
    return d[keep].dropna().reset_index(drop=True)

def impute_zero_values(df, columns):
    """
    Replaces 0 values in specified numeric columns with the mean of non-zero values.
    """
    df_clean = df.copy()
    for col in columns:
        mean_val = df_clean.loc[df_clean[col] != 0, col].mean()
        df_clean[col] = df_clean[col].replace(0, mean_val)
    return df_clean
