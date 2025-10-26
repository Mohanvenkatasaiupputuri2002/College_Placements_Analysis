import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss


def plot_relational_plot(df):
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='IQ', y='CGPA', hue='Placement', palette='coolwarm')
    plt.title('Relational Plot: IQ vs CGPA by Placement')
    plt.xlabel('IQ')
    plt.ylabel('CGPA')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('relational_plot.png')
    plt.show()
    plt.close()
    return


def plot_categorical_plot(df):
    
    plt.figure(figsize=(6, 6))
    df['Placement'].value_counts().plot(kind='bar', color=['lightgreen', 'lightcoral'])
    plt.title('Categorical Plot: Placement Status Distribution')
    plt.xlabel('Placement Status')
    plt.ylabel('Number of Students')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('categorical_plot.png')
    plt.show()
    plt.close()
    return


def plot_statistical_plot(df):
   
    # Pie chart
    plt.figure(figsize=(6, 6))
    df['Internship_Experience'].value_counts().plot(
        kind='pie', autopct='%1.1f%%', colors=['skyblue', 'orange']
    )
    plt.title('Internship Experience (Pie Chart)')
    plt.ylabel('')
    plt.tight_layout()
    plt.savefig('pie_chart.png')
    plt.show()
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap: College Placement Dataset')
    plt.tight_layout()
    plt.savefig('statistical_plot.png')
    plt.show()
    plt.close()
    return


def statistical_analysis(df, col: str):
   
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col], nan_policy='omit')
    excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')
    return mean, stddev, skew, excess_kurtosis



def preprocessing(df):
    
    print("Data Information:")
    print(df.info())
    print("\nData Head:\n", df.head())
    print("\nData Tail:\n", df.tail())
    print("\nDescriptive Statistics:\n", df.describe())
    print("\nCorrelation Matrix:\n", df.corr(numeric_only=True))
    return df


def writing(moments, col):
    
    mean, stddev, skew, excess_kurtosis = moments
    print(f'\nFor the attribute "{col}":')
    print(f'Mean = {mean:.2f}, Standard Deviation = {stddev:.2f}, '
          f'Skewness = {skew:.2f}, Excess Kurtosis = {excess_kurtosis:.2f}.')

    # Interpretation of skewness and kurtosis
    if skew > 0.5:
        skew_type = 'right-skewed'
    elif skew < -0.5:
        skew_type = 'left-skewed'
    else:
        skew_type = 'approximately symmetric'

    if excess_kurtosis > 2:
        kurt_type = 'leptokurtic'
    elif excess_kurtosis < -2:
        kurt_type = 'platykurtic'
    else:
        kurt_type = 'mesokurtic'

    print(f'The data is {skew_type} and {kurt_type}.\n')


def main():
    
    df = pd.read_csv('data.csv')
    df = preprocessing(df)

    col = 'CGPA'  # You can change this to analyze other numerical columns
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)


if __name__ == '__main__':
    main()
