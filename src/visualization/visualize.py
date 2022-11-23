import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent.as_posix()
FIG_DIR = f'{ROOT_DIR}/reports/figures/'

plt.style.use("fivethirtyeight")


def histogram(df,col,output_prefix,bins = 40):
    fig, ax = plt.subplots()
    ax.hist(df[col], bins = bins,alpha=0.75)
    title = col.replace('_',' ').title()
    ax.set_title(title,fontsize=18.5, fontweight='semibold', color='#414141')
    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')
    ax.axhline(y=0.1, color='#414141', linewidth=2.5)
    fig.tight_layout()
    plt.show()
    plt.savefig(f'{FIG_DIR}{output_prefix}_hist.png')

def box_plot(df,col,output_prefix):
    title = col.replace('_',' ').title()
    plt.style.use("fivethirtyeight")
    five_thirty_eight = [
    "#30a2da",
    "#fc4f30",
    "#e5ae38",
    "#6d904f",
    "#8b8b8b",
    ]
    sns.set(
        #style="dark",                   # The 'ticks' style
        rc={"figure.figsize": (6, 9)})
    sns.set_palette(five_thirty_eight)
    b = sns.boxplot(y=df[col])
    b.set_ylabel(title, fontsize = 14)
    # Set the x axis label and font size
    b.set_xlabel("All Columns", fontsize = 14)
    # Set the plot title with the pval variable and font size
    b.set_title(f'Variability Analysis: {title}',fontsize=18.5, fontweight='semibold', color='#414141')
    b.figure.savefig(f"{FIG_DIR}/{output_prefix}_boxplot.png")