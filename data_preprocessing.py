import pandas as pd


# read data onto pandas dataframe
def load_data(filename):
    data = pd.read_csv(filename)
    return data


# Transform attribute values to binary
def boolean_to_binary(data):
    for i in range(data.index.size):
        if data.loc[i, "Civil_Twilight"] == "Night":
            data.loc[i, "Civil_Twilight"] = 0
        else:
            data.loc[i, "Civil_Twilight"] = 1
        if pd.isna(data.loc[i, "Precipitation(in)"]):
            data.loc[i, "Precipitation(in)"] = 0
        if pd.isna(data.loc[i, "Wind_Speed(mph)"]):
            data.loc[i, "Wind_Speed(mph)"] = 0

    return data.dropna()


# This section is to do some of the heavy pre-processing which takes a long time.
# In order to make things more efficient given the size of the dataset we store
# the modified data to be read later
if __name__ == "__main__":
    dataset = boolean_to_binary(load_data("US_Accidents_Dec20.csv"))
    dataset.to_csv("modified1.csv", index=False)
