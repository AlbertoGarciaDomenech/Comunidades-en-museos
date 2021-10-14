import numpy as np
from PIL import Image
from setup import PATHS
from skimage.io import imread
from collections import Counter
from PIL.ImageColor import getrgb 
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from cluster_caracterization import clustersInformation



plt.style.use('seaborn')

def histogramFromDict(ax, data, color, title):
    ax.barh(np.arange(len(data.keys())), data.values(), color=color, align='center')
    ax.set_yticks(np.arange(len(data.keys())))
    ax.set_xticks(np.arange(.0,1.1,.25))
    ax.set_yticklabels(data.keys(), fontsize=17)
    ax.set_xticklabels([str(i) + '%' for i in np.arange(0,101,25)], fontsize=20)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=35)

def colorFrame(image, color='#d9cda2'):
    size = max(image.size)
    square = Image.new('RGB', (size, size), getrgb(color))
    square.paste(image, (int((size - image.size[0]) / 2), int((size - image.size[1]) / 2)))
    return square

def showImage(ax, title, url, color):
    ax.imshow(colorFrame(Image.fromarray(np.uint8(imread(url))).convert('RGB'), color))
    ax.set_title(title, fontsize=18, color=color, fontweight='semibold')
    ax.set_xticks([])
    ax.set_yticks([])

def displayList(ax, title, list, color):
    ax.text(0, 1, title, color=color, fontsize=35, fontweight='bold')
    for i, item in enumerate(list):
        ax.text(.1, .8 - ((.8/len(list)) * i), str(i+1) + ' - ' + item, color=color, fontsize=25)
    ax.axis('off')

def clusterInfographic(info, title):

    pp = PdfPages(PATHS['CLUSTER_REPORT'] + title.replace(' ','') + '.pdf')

    for cluster, info in info['Clusters'].items():

        fig, ax = plt.subplots(figsize=(20.0, 20.0) , nrows=4, ncols=3)
        fig.suptitle(cluster + ' information (Size = ' + str(info['Size']) + ')', fontsize=46)

        # Demographic information
        histogramFromDict(ax[0][0], info['Demographics']['age'], '#a4c13e', 'Age')
        histogramFromDict(ax[0][1], info['Demographics']['country'], '#f3e14b', 'Nationality')
        histogramFromDict(ax[0][2], info['Demographics']['gender'], '#f1aa4f', 'Gender')

        # Top artworks
        for i, artwork in enumerate(info['Artistic']['Most valued']):
            showImage(ax[1][i], 
                      '#' + str(i+1) + ' Positive:\n'+ artwork['Title'] + '\n[Pos:' + str(artwork['Positive']) + ' | Neg:' + str(artwork['Negative']) + ']', 
                      artwork['Image'], 
                      '#3ec19d')

        # Bottom artworks
        for i, artwork in enumerate(info['Artistic']['Least valued']):
            showImage(ax[2][i], 
                      '#' + str(i+1) + ' Negative:\n'+ artwork['Title'] + '\n[Pos:' + str(artwork['Positive']) + ' | Neg:' + str(artwork['Negative']) + ']', 
                      artwork['Image'], 
                      '#c13e62')

        for i in range(len(ax[0])):
            ax[1][i].axis('off')
            ax[2][i].axis('off')

        # More info
        displayList(ax[3][0], "Positive valued artists", info['Artistic']['Popular artists'], color='#3ec19d')
        displayList(ax[3][1], "Negative valued artists", info['Artistic']['Unpopular artists'], color='#c13e62')
        ax[3][2].text(0, 1, "Top Category", color='#56420e', fontsize=30, fontweight='bold')
        ax[3][2].text(.1, .8, info['Artistic']['Top Category'], color='#56420e', fontsize=35)
        
        topEmotions = Counter(info["Emotions"]).most_common(2)
        ax[3][2].text(0, .6, "Top Emotions", color='#b810c4', fontsize=30, fontweight='bold')
        ax[3][2].text(.1, .4, topEmotions[0][0].upper(), color='#b810c4', fontsize=30)
        ax[3][2].text(.1, .2, topEmotions[1][0].upper(), color='#b810c4', fontsize=30)
        ax[3][2].axis('off')

        fig.tight_layout()
        fig.savefig(pp, format='pdf')
        plt.close('all')
    
    pp.close()