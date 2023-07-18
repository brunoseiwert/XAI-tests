import pandas as pd

def decode_instance(instance, categorical_names):
    decoded_instance = instance.copy().astype(object)
    for (index, names) in categorical_names.items():
        decoded_instance[index] = decoded_instance[index]
        decoded_instance[index] = names[int(instance[index])]
    return decoded_instance

def encode_instance(instance, categorical_names):
    encoded_instance = instance.copy().astype(object)
    for (index, names) in categorical_names.items():
        for (i, name) in enumerate(names):
            if name == encoded_instance[index]:
                encoded_instance[index] = i
    return encoded_instance

def encode_dataframe(df, categorical_names):
    new_df = pd.DataFrame(columns=df.columns, index=df.index, dtype='object')
    for i in range(len(df)):
        new_df.iloc[i] = encode_instance(df.iloc[i], categorical_names)
    return new_df

def decode_dataframe(df, categorical_names):
    new_df = pd.DataFrame(columns=df.columns, index=df.index, dtype='object')
    for i in range(len(df)):
        new_df.iloc[i] = decode_instance(df.iloc[i], categorical_names)
    return new_df