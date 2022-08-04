# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from textwrap import dedent as d
import pickle
import numpy as np
import plotly
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.figure_factory as ff
from dash.dependencies import Input, Output, State
import json
import pandas as pd
import time
from itertools import zip_longest

colormap1 = [[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'], [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'], [0.6666666666666666, 'rgb(171,217,233)'], [0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'], [1.0, 'rgb(49,54,149)']]
colormap2 = [[0.0,'rgb(85, 156, 178)'], [0.20, 'rgb(245, 238, 248)'], [0.40, 'rgb(244, 236, 247)'], [0.60, 'rgb(253, 237, 236)'], [0.80, 'rgb(245, 183, 177)'], [1.0, 'rgb(195, 115, 99)']]

def grouper(n, iterable, padvalue=0):
  "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
  return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def get_normalized_weights(weight_list):
    filtered_list = list(filter(lambda x: x[0] != '<pad>', weight_list))
    total_weight = sum([elem[1] for elem in filtered_list])
    return_list = list(map(lambda x: (x[0], x[1]/total_weight), filtered_list))
    #print(return_list)
    return(return_list)

def make_plots(weight_list):
    colorscale = [[0, '#ecbfe0'], [1, '#66475e']]
    font_colors = ['#3c3636','#efecee']
    #print("Attention values ", weight_list)
    annotation_text, vals = zip(*weight_list)
    #vals = [list(vals[0:10]), list(vals[10:20]), list(vals[20:])][::-1]
    vals = list(grouper(5, vals, padvalue= 0))[::-1]
    #annotation_text = [list(annotation_text[0:10]), list(annotation_text[10:20]), list(annotation_text[20:])][::-1]
    annotation_text = list(grouper(5, annotation_text, padvalue= 'PAD'))[::-1]
    #print(vals)
    #print(annotation_text)
    layout = go.Layout(
        paper_bgcolor='rgba(255,255,255,0)',
        #plot_bgcolor='rgba(0,0,0,0)'
    )
    fig = ff.create_annotated_heatmap(vals, colorscale=colorscale, annotation_text=annotation_text, font_colors=font_colors)
    fig.layout.title = 'Weights'
    fig.layout.paper_bgcolor = 'rgba(255,255,255,0)'
    return(fig)

def create_dummy_attention(attention_weights =[('No',1),('document',1),('is',2),('selected',1),('currently',5)]):
    fig = make_plots(attention_weights)
    return(fig)

def create_attention(weight_list):
    fig = make_plots(weight_list)
    return(fig)

class Training():

    def __init__(self, training_file, header):
        self.filename = training_file
        self.header = header
        self.read_training_file()

    def read_training_file(self):
        self.df = pd.read_csv(self.filename, header=self.header)
        self.df.columns = ['record_id','text','id','label']
        self.df.set_index('id',inplace=True)
        print("Training file", self.df.head())


class Doc(Training):

    def __init__(self, training_file, header=None):
        # This is the test set that has the four columns record_id, text, id and label. id is the index of this dataframe
        # This contains the text that will be displayed in the right hand size when a point is selected. This is indexed through id
        super().__init__(training_file, header)
        self.classifier_df = pd.DataFrame()

    def load_data_file(self, filename):
        # Load the visualization files
        infile = open(filename,'rb')
        new_dict = pickle.load(infile)
        infile.close()

        if(isinstance(new_dict['text'][0], list)):
            one_dim_list_text = [ elem[0] for elem in new_dict['text']]
            new_dict['text'] = one_dim_list_text

        print("Vis file keys ",new_dict.keys())

        self.data_dict = new_dict
        self.X_reduced = new_dict['X']

        self.classifier_df['text'] = new_dict['text']
        self.label = new_dict['label']
        print("Number of entries in Vis file ",len(self.label))
        self.score = new_dict['score']
        self.score_tag = new_dict['score_tag']
        self.title = new_dict['title']
        self.tweet_id = new_dict['tweet_id']
        print(self.tweet_id.head())
        self.attention_weights = new_dict.get('attention_weights', pd.Series())

    def get_scatterplot_fig(self):

        trace1 = go.Scattergl(
        x = self.X_reduced[:,0],
        y = self.X_reduced[:,1],
        mode = 'markers',
        text = self.classifier_df['text'],
        marker = dict(
            color = np.array(self.label),
            opacity = self.score,
            size = 15,
            colorscale = colormap2,
            line = dict(
                  color = '#F5B7B1',
                  width = self.score_tag)
            ),
        name = 'embeddings '
        )

        data = [trace1]
        layout = go.Layout(
        title=self.title,
        xaxis=dict(
            title='x component',
        ),
        yaxis=dict(
            title='y component',
            ),
        plot_bgcolor= '#202020',
        paper_bgcolor= '#202020',
        font= dict(
            color= 'gray'
        )
        )
        fig = dict(data=data,layout=layout)
        return(fig)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

styles = {
    'pre': {
        'overflowX': 'auto',
        'color': 'gray',
        'max-height': '600px',
        'whitespace': 'pre-wrap',
        'word-wrap' : 'break-word',
        'padding': '10px',
        'font-family': 'courier new',
        'font-size': '28'
    },
    'color': 'gray'
}

# --------------------------- Initialize application ----------------------------- #
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
# ---------- RNN example ---------- #
# TEST_FILE = '../Rnn/Training_data.csv'
# PCA_file = '../Rnn/Vis_data_pca.txt'
# TSNE_file = '../Rnn/Vis_data_tsne.txt'
# pca_obj = Doc(TEST_FILE, header=0)
# tsne_obj = Doc(TEST_FILE, header=0)
# pca_obj.load_data_file(PCA_file)
# tsne_obj.load_data_file(TSNE_file)

# --------- LSTM attention pytorch --------- #
# TEST_FILE = '../Pytorch/Training_data.csv'
# PCA_file = '../Pytorch/Vis_data_pca.txt'
# TSNE_file = '../Pytorch/Vis_data_tsne.txt'
# pca_obj = Doc(TEST_FILE, header=0)
# tsne_obj = Doc(TEST_FILE, header=0)
# pca_obj.load_data_file(PCA_file)
# tsne_obj.load_data_file(TSNE_file)

# --------- Pytorch_curated --------- #
TEST_FILE = './results/Pytorch_curated/Test.csv'
PCA_file = './results/Pytorch_curated/Vis_data_pca.txt'
TSNE_file = './results/Pytorch_curated/Vis_data_tsne.txt'
pca_obj = Doc(TEST_FILE, header=0)
tsne_obj = Doc(TEST_FILE, header=0)
pca_obj.load_data_file(PCA_file)
tsne_obj.load_data_file(TSNE_file)


# --------- attn_pytorch_opt --------- #
#TEST_FILE = './results/attn_pytorch_opt/Training_data.csv'
#PCA_file = './results/attn_pytorch_opt/Vis_data_pca.txt'
#TSNE_file = './results/attn_pytorch_opt/Vis_data_tsne.txt'
#pca_obj = Doc(TEST_FILE, header=0)
#tsne_obj = Doc(TEST_FILE, header=0)
#pca_obj.load_data_file(PCA_file)
#tsne_obj.load_data_file(TSNE_file)


# --------- Elmo_curated --------- #
# TEST_FILE = './results/Elmo_curated/Training_data.csv'
# PCA_file = './results/Elmo_curated/Vis_data_pca.txt'
# TSNE_file = './results/Elmo_curated/Vis_data_tsne.txt'
# pca_obj = Doc(TEST_FILE, header=0)
# tsne_obj = Doc(TEST_FILE, header=0)
# pca_obj.load_data_file(PCA_file)
# tsne_obj.load_data_file(TSNE_file)

# --------- Elmo attention ---------- #
#TEST_FILE = '../Elmo/Training_data.csv'
#PCA_file = '../Elmo/Elmo_attention_results/Vis_data_pca.txt'
#TSNE_file = '../Elmo/Elmo_attention_results/Vis_data_tsne.txt'
#pca_obj = Doc(TEST_FILE, header=0)
#tsne_obj = Doc(TEST_FILE, header=0)
#pca_obj.load_data_file(PCA_file)
#tsne_obj.load_data_file(TSNE_file)

#fig_returned = pca_obj.get_scatterplot_fig()
fig_returned = create_dummy_attention()
fig_returned2 = tsne_obj.get_scatterplot_fig()

selected_point = None
selected_point_index = None

colors = {
  'background': '#111111',
  'text': '#7FDBFF'
}

# ------------------------- Application layout information ----------------------- #

existing_style = {'overflowX': 'auto', 'color': 'gray', 'max-height': '600px', 'whitespace': 'pre-wrap', 'word-wrap': 'break-word', 'padding': '10px', 'font-family': 'courier new', 'font-size': '28'}

app.layout = html.Div(style={'backgroundColor': '#202020'}, children=[

  html.H1(
      children='Visualization for stance detection',
      style={
          'textAlign': 'center',
          'color': 'gray',
          'font-family': 'monospace'
      }
  ),

  dcc.RadioItems(
    id='radio-items',
    style={
        'textAlign': 'center',
        'color': 'gray',
        'font-family': 'monospace'
    },
    options = [
        {'label': 'TSNE', 'value': 'tsne'},
        {'label': 'PCA', 'value': 'pca'}
        ],
    value = 'tsne',
    labelStyle={'display': 'inline-block'}
    ),


  html.Div([
  dcc.Graph(
      id='tsne-graph',
      figure=fig_returned2
  ),


  dcc.Graph(
      id='attention-graph',
      figure=fig_returned
  )
  ]
  ,
  style={'width': '65%', 'display': 'inline-block'}
  ),


 html.Div([
 html.H5(
     children='Select a document in either \n visualization by clicking on it, this toggles the affiliation',
     style={
         'textAlign': 'center',
         'color': 'gray',
         'font-family': 'courier new'
     }
 ),

 html.Pre(id='click-data', style=existing_style),
 #html.Pre(id='click-data', style={'font-family': 'courier new', 'font-size': '5'}),

 html.Button(id='toggle-button', n_clicks=0, children='Click here to toggle affiliation'),

 html.Pre(id='toggle-data', style=styles['pre']),

 html.Button(id='save-button', n_clicks=0, children='Export current state'),

 html.Pre(id='save-data', style=styles['pre']),

 ]
 ,
 style={'width': '35%', 'display': 'inline-block', 'float': 'right', 'white-space': 'pre-wrap', 'word-break': 'keep-all'}
 )

])

print("STYLES ",styles['pre'])

@app.callback(
    Output('click-data', 'children'),
    [Input('tsne-graph', 'clickData')])
def display_click_data(clickData):
    global selected_point, selected_point_index
    if(clickData):
        elem = clickData.get('points')[0]
        c_text = clickData.get('points')[0].get('text')
        c_label = 'conservative' if (elem.get('marker.color') == 1) else 'liberal'
        c_classification = 'MISMATCH' if (elem.get('marker.line.width') == 5) else 'Matches corpus affiliation'
        c_classification_strength = elem.get('marker.opacity')

        #print("TSNE TWEET ID ",tsne_obj.tweet_id)
        #print("DONE")
        c_tweet_id = tsne_obj.tweet_id[elem.get('pointNumber')]
        selected_point = c_tweet_id
        selected_point_index = elem.get('pointNumber')
        #print(clickData)
        #print(elem)
        #print(c_text,c_label)
        return_text = 'Selected text: \n' + c_text + '\n\n' + 'Affiliation in corpus: ' + c_label + '\n\n' + \
        'Classification affiliation: ' + c_classification + '\n\n' + \
        'Normalized classification strength: ' + str(c_classification_strength) + '\n\n' + \
        'Tweet ID: ' + str(c_tweet_id)

        return(return_text)
    else:
        return('No point selected \n\n\n\n\n\n\n\n\n\n\n')


@app.callback(
    Output('attention-graph','figure'),
    [Input('tsne-graph','clickData')])
def update_attention(clickData):
    if(clickData):
        elem = clickData.get('points')[0]
        elem_index = elem.get('pointNumber')
        c_tweet_id = tsne_obj.tweet_id[elem_index]
        #weight_list = tsne_obj.df.ix[]
        print("Element selected for pca graph",c_tweet_id, elem_index)
        if(not tsne_obj.attention_weights.empty):
            fig_returned = create_attention(tsne_obj.attention_weights[elem_index])
        else:
            fig_returned = create_dummy_attention(attention_weights =[('No',1),('attention',1),('weights',2),('are',1),('available',5)])
    else:
        fig_returned = create_dummy_attention()
    return(fig_returned)


@app.callback(
    Output('toggle-data', 'children'),
    [Input('toggle-button', 'n_clicks')])
def display_click_data(n_clicks):
    global tsne_obj

    text = "Toggled point " + str(selected_point)
    print("Toggled point ",selected_point)
    print("SELECTED POINT ", selected_point in list(tsne_obj.tweet_id.values))

    if(selected_point):
        print("Inside selected point")
        text = text + " from label " + str(tsne_obj.df.ix[int(selected_point)].label)
        print("Affiliation in training file", tsne_obj.df.ix[int(selected_point)].text, tsne_obj.df.ix[int(selected_point)].label)
        tsne_obj.df.ix[int(selected_point),'label'] = int(abs(1 - tsne_obj.df.ix[int(selected_point)].label))
        print("Elem after toggling ",tsne_obj.df.ix[int(selected_point)].label)
        text = text + " to label " + str(tsne_obj.df.ix[int(selected_point)].label)

    if(n_clicks == 0):
        text = " "
    return(text)


@app.callback(
    Output('save-data', 'children'),
    [Input('save-button', 'n_clicks')])
def save_data_callback(n_clicks):
    global tsne_obj
    print(tsne_obj.df.columns)
    tsne_obj.df.to_csv(header=False, columns=['text','label'], path_or_buf='Exported.csv')
    text = "Exported state at time " + time.ctime()

    if(n_clicks == 0):
        text = " "
    return(text)

@app.callback(Output('tsne-graph', 'figure'),
    [Input('radio-items', 'value')])
def make_scatter_chart(value):
    if(value == 'pca'):
        pca_obj = Doc(TEST_FILE)
        pca_obj.load_data_file(PCA_file)
        fig_returned2 = pca_obj.get_scatterplot_fig()
    else:
        tsne_obj = Doc(TEST_FILE)
        tsne_obj.load_data_file(TSNE_file)
        fig_returned2 = tsne_obj.get_scatterplot_fig()

    return(fig_returned2)



if __name__ == '__main__':
  app.run_server(debug=True)
