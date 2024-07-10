import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('/workspace/boilerplate-medical-data-visualizer/medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2) > 25).astype(int)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'alco', 'smoke', 'active', 'overweight', ])


    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'])['value'].count().rename('total').reset_index()
    

    # 7
    df_cat.columns = ['cardio', 'variable', 'value', 'total']

    # 8
    fig = sns.catplot(data=df_cat, x='variable', y='total', hue='value', col='cardio', order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], kind='bar', errorbar=None).fig


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[(df.ap_lo <= df.ap_hi) & (
        df['height'] >= df['height'].quantile(0.025)) & (df['height'] <= df['height'].quantile(.975)) & (df['weight'] >= df['weight'].quantile(.025)) & (df.weight <= df.weight.quantile(.975))]


    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))




    # 14
    fig, ax = plt.subplots(figsize=(10,8))

    # 15
    sns.heatmap(corr, mask = mask, annot = True, cmap = 'coolwarm',fmt = '.1f', ax=ax)


    # 16
    fig.savefig('heatmap.png')
    return fig
