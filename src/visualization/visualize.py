import seaborn as sns
from matplotlib import pyplot as plt
from pathlib import Path
import re
ROOT_DIR = Path(__file__).parent.parent.parent.as_posix()
FIG_DIR = f'{ROOT_DIR}/reports/figures/'

plt.style.use("fivethirtyeight")

# TODO image not outputting to folder and bar cut off

def histogram(df,col,output_prefix,fig_dir,subtitle = '', title = None, show_graph = True, bins = 40, percent_col=None,log_scale = None):
    fig, ax = plt.subplots()
    ax.hist(df[col], bins = bins,alpha=0.75)
    if title == None:
        title = col.replace('_',' ').title()
    # ax.set_title(f'{title} ',fontsize=18.5, fontweight='semibold', color='#414141')
    # prefix = re.search('(.*?-.*?)-', output_prefix).group(1)
    prefix = output_prefix.split("--")[0]
    if title != None:
        output_prefix = f'{prefix}_{title.replace(" ","_")}_{subtitle.replace(" ","_")}'.lower()
    plt.figtext(0.05, 1.05, title,fontsize=18.5, fontweight='semibold', color='#414141')
    plt.figtext(0.05, 1, subtitle,fontsize=16.5, color='#414141')
    ax.set_xlabel(col.replace('_',' ').title())
    ax.set_ylabel('Frequency')
    if str(percent_col).lower() == 'x':
        vals = ax.get_xticks()
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    if str(percent_col).lower() == 'y':
        vals = ax.get_xticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    if log_scale == True:
        plt.gca().set_yscale("log")
        ax.bar_label(ax.containers[0], label_type='edge')
        plt.gca().axes.yaxis.set_ticklabels([])

        ax.yaxis.set_ticks_position('none')
        ax.set_ylabel('Log Frequency')
        ax.axhline(y=0, color='#414141', linewidth=2.5)
    else:
        ax.axhline(y=0.1, color='#414141', linewidth=2.5)
    fig.tight_layout()
    print(f'Image found in the "reports/figures" folder saved under {output_prefix}_{col}_hist.png')
    plt.savefig(f'{fig_dir}{output_prefix}_{col}_hist.png',bbox_inches='tight')
    # if show_graph == True:
    #     plt.show()

def line_plot(history, fig_dir, output_prefix, subtitle, title = 'Model Evaluation'):
    import pandas as pd
    s = pd.DataFrame(history.history)
    s['epochs'] = range(1,len(s)+1)
    fig, ax = plt.subplots()
    plt.plot(s['epochs'],s['loss'],label = "training loss")
    plt.plot(s['epochs'],s['val_loss'],label = "validation loss")
    # plt.figtext(0.05, 1.05, title,fontsize=18.5, fontweight='semibold', color='#414141')
    # plt.figtext(0.05, 1, subtitle,fontsize=16.5, color='#414141')

    plt.figtext(0.05, 1.05, title.replace('_',' ').title(),fontsize=18.5, fontweight='semibold', color='#414141')
    plt.figtext(0.05, 1, subtitle.replace('_',' ').title(),fontsize=16.5, color='#414141')
    ax.set_xlabel("Epochs".title())
    ax.set_ylabel('MSE')
    # plt.boxplot(df[col],patch_artist=True)
    plt.legend()
    fig.tight_layout()
    print(f'Image found in the "reports/figures" folder saved under {output_prefix}_{title.lower().replace(" ","_")}_{subtitle.replace(" ","_").lower()}_lineplot.png')
    plt.savefig(f'{fig_dir}{output_prefix}_{title.lower().replace(" ","_")}_{subtitle.replace(" ","_").lower()}_lineplot.jpg',bbox_inches='tight')
    

def box_plot(df,col,subtitle, fig_dir, output_prefix, title = None, show_graph = True,percent_col = None):
    if title == None:
        title = col.replace('_',' ').title()
    # prefix = re.search('(.*?-.*?)-', output_prefix).group(1)
    prefix = output_prefix.split("--")[0]
    if title != None:
        output_prefix = f'{prefix}_{title.replace(" ","_")}_{subtitle.replace(" ","_")}'.lower()
    fig, ax = plt.subplots()
    plt.figtext(0.05, 1.05, title,fontsize=18.5, fontweight='semibold', color='#414141')
    plt.figtext(0.05, 1, subtitle,fontsize=16.5, color='#414141')
    ax.set_xlabel(col.replace('_',' ').title())
    ax.set_ylabel('Frequency')
    plt.boxplot(df[col],patch_artist=True)
    if str(percent_col).lower() == 'x':
        vals = ax.get_xticks()
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    if str(percent_col).lower() == 'y':
        vals = ax.get_xticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    fig.tight_layout()
    print(f'Image found in the "reports/figures" folder saved under {output_prefix}_{col}_boxplot.png')
    plt.savefig(f"{fig_dir}{output_prefix}_{col}_boxplot.jpg",bbox_inches='tight')
    # if show_graph == True:
    #     plt.show()
    
def bar_plot(df,x,y,title,subtitle,fig_dir,output_prefix,n_cols = 12,show_graph = True, width = None, height = None,percent_col = None):
    fig, ax = plt.subplots()
    # prefix = re.search('(.*?-.*?)-', output_prefix).group(1)
    prefix = output_prefix.split("--")[0]
    if title != None:
        output_prefix = f'{prefix}_{title.replace(" ","_")}_{subtitle.replace(" ","_")}'.lower()
    df = df.nlargest(n_cols,y)
    df = df.sort_values(y, ascending = True)
    ax.barh(df[x],df[y],alpha=0.75)
    plt.figtext(0.05, 1.05, title,fontsize=18.5, fontweight='semibold', color='#414141')
    plt.figtext(0.05, 1, subtitle,fontsize=16.5, color='#414141')
    ax.set_xlabel(y.replace('_',' ').title())
    ax.set_ylabel(x.replace('_',' ').title())
    if width != None and height != None:
        fig.set_size_inches(width, height)
    if str(percent_col).lower() == 'x':
        vals = ax.get_xticks()
        ax.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    if str(percent_col).lower() == 'y':
        vals = ax.get_xticks()
        ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    fig.tight_layout()
    
    print(f'Image found in the "reports/figures" folder saved under {output_prefix}_{x}_{y}_barchart.png')
    plt.savefig(f"{fig_dir}{output_prefix}_{x}_{y}_barchart.jpg",bbox_inches='tight')
    # if show_graph == True:
    #     plt.show()