import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
ROOT_DIR = Path(__file__).parent.parent.parent.as_posix()
FIG_DIR = f'{ROOT_DIR}/reports/figures/'

plt.style.use("fivethirtyeight")

# TODO image not outputting to folder and bar cut off

def histogram(df,col,output_prefix,fig_dir,subtitle = '', show_graph = True, bins = 40):
    fig, ax = plt.subplots()
    ax.hist(df[col], bins = bins,alpha=0.75)
    title = col.replace('_',' ').title()
    # ax.set_title(f'{title} ',fontsize=18.5, fontweight='semibold', color='#414141')
    plt.figtext(0.05, 1.05, title,fontsize=18.5, fontweight='semibold', color='#414141')
    plt.figtext(0.05, 1, subtitle,fontsize=16.5, color='#414141')
    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')
    ax.axhline(y=0.1, color='#414141', linewidth=2.5)
    fig.tight_layout()
    if show_graph == True:
        plt.show()
    plt.savefig(f'{fig_dir}{output_prefix}_{col}_hist.png')

def box_plot(df,col,subtitle, fig_dir, output_prefix, show_graph = True,):
    title = col.replace('_',' ').title()
    fig, ax = plt.subplots()
    plt.figtext(0.05, 1.05, title,fontsize=18.5, fontweight='semibold', color='#414141')
    plt.figtext(0.05, 1, subtitle,fontsize=16.5, color='#414141')
    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')
    plt.boxplot(df[col],patch_artist=True)
    fig.tight_layout()
    if show_graph == True:
        plt.show()
    plt.savefig(f"{fig_dir}{output_prefix}_{col}_boxplot.png")

def bar_plot(df,x,y,title,subtitle,fig_dir,output_prefix,n_cols = 12,show_graph = True):
    fig, ax = plt.subplots()
    df = df.nlargest(n_cols,y)
    df = df.sort_values(y, ascending = True)
    ax.barh(df[x],df[y],alpha=0.75)
    plt.figtext(0.05, 1.05, title,fontsize=18.5, fontweight='semibold', color='#414141')
    plt.figtext(0.05, 1, subtitle,fontsize=16.5, color='#414141')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    if show_graph == True:
        plt.show()
    plt.savefig(f"{fig_dir}{output_prefix}_{x}_{y}_barchart.png")