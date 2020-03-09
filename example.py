from nn_ols import nn_ols
import pandas as pd

df = pd.read_csv("ingredients_matrix.csv")

for ingredient in df.columns[1:]:
    x = df.drop(['Row Labels', ingredient], axis=1,)
    y = df[ingredient]
    trained_model = nn_ols(x,y)
    print(trained_model)
    print(f"Number of non-zero coefficients: {trained_model.non_zero_ingredients}")
    print(f"Weights: {trained_model.weights}")
