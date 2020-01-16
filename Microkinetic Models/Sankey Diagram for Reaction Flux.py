
# coding: utf-8

# In[ ]:


from ipysankeywidget import SankeyWidget
from ipywidgets import Layout
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np


# In[ ]:


layout = Layout(width="300", height="200")
def sankey(margin_top=10, **value):
    """Show SankeyWidget with default values for size and margins"""
    return SankeyWidget(layout=layout,
                        margins=dict(top=margin_top, bottom=0, left=30, right=60),
                        **value)


# In[ ]:


#Pt(211) at 500 K

#Rearcions
#0 O2 + 2* = 2O*
#1 NH3 + * = NH3*
#2 NH3* + O* = NH2* + OH*
#3 NH2* + O* = NH* + OH*
#4 NH* + O* = N* + OH*
#5 NH3* + OH* = NH2* + H2O*
#6 NH2* + OH* = NH* + H2O*
#7 NH* + OH* = N* + H2O*
#8 OH* + OH* = O* + H2O*
#9 H2O + * = H2O*
#10 N* + N* = N2 + *
#11 N* + O* = NO* + *
#12 NO* = NO + *
#13 N* + NO* =N2O*
#14 N2O* = N2O + *
#15 NH3* + * = NH2* + H*
#16 NH2* + * = NH* + H*
#17 NH* + * = N* + H*


rate = [0.005296340606381573,
        0.007057013086015473,
        0.007056393131952513,
        1.2805229581623417e-05,
        0.007043471964008674,
        6.199275870618408e-07,
        0.007044207858129278,
        1.8098114960731583e-05,
        0.003745609401993047,
        -0.01058551962898946,
        0.003521444576068399,
        7.161583737431936e-06,
        1.9923359372324168e-07,
        6.962350143708696e-06,
        6.962350143708694e-06,
        2.647751962138639e-11,
        -1.6938070175571887e-12,
        -4.556992953049709e-06,
        0.00021845870547452023,
        -0.00022301567364385732]


# In[ ]:


links = [
    {'source': r'NH3*', 'target': 'OH*', 'value': rate[5], 'color': 'lightgrey'},
    {'source': r'NH3*', 'target': 'O*', 'value': rate[2], 'color': 'lightsteelblue'},
    {'source': r'NH3*', 'target': '*', 'value': rate[15], 'color': 'lightcoral'},
    {'source': 'OH*', 'target': 'NH2*', 'value': rate[5], 'color': 'lightgrey'},
    {'source': 'O*', 'target': 'NH2*', 'value': rate[2], 'color': 'lightsteelblue'},
    {'source': '*', 'target': 'NH2*', 'value': rate[15], 'color': 'lightcoral'},
    {'source': 'NH2*', 'target': 'OH* ', 'value': rate[6], 'color': 'lightgrey'},
    {'source': 'NH2*', 'target': 'O* ', 'value': rate[3], 'color': 'lightsteelblue'}, 
    {'source': 'NH2*', 'target': '* ', 'value': rate[16], 'color': 'lightcoral'},     
    {'source': 'OH* ', 'target': 'NH*', 'value': rate[6], 'color': 'lightgrey'},
    {'source': 'O* ', 'target': 'NH*', 'value': rate[3], 'color': 'lightsteelblue'},
    {'source': '* ', 'target': 'NH*', 'value': rate[16], 'color': 'lightcoral'},
    {'source': 'NH*', 'target': 'OH*  ', 'value': rate[7], 'color': 'lightgrey'},
    {'source': 'NH*', 'target': 'O*  ', 'value': rate[4], 'color': 'lightsteelblue'},
    {'source': 'NH*', 'target': '*  ', 'value': rate[17], 'color': 'lightcoral'},     
    {'source': 'OH*  ', 'target': 'N*', 'value': rate[7], 'color': 'lightgrey'},
    {'source': 'O*  ', 'target': 'N*', 'value': rate[4], 'color': 'lightsteelblue'}, 
    {'source': '*  ', 'target': 'N*', 'value': rate[17], 'color': 'lightcoral'}, 
    {'source': 'N*', 'target': 'NO*', 'value': rate[11], 'color': 'orange'},
    {'source': 'N*', 'target': 'N2', 'value': rate[10]*2, 'color': 'steelblue'}, 
    {'source': 'N*', 'target': 'N2O', 'value': rate[14], 'color': 'violet'},
    {'source': 'NO*', 'target': 'N2O', 'value': rate[13], 'color': 'violet'},
    {'source': 'NO*', 'target': 'NO', 'value': rate[12], 'color': 'orange'},
]

rank_sets = [
    { 'type': 'same', 'nodes': ['N2','N2O','NO'] },
]
Pt211_500network = SankeyWidget(links=links, margins=dict(top=150, bottom=150, left=150, right=150),rank_sets=rank_sets)


# In[ ]:


Pt211_500network


# In[ ]:


# Pt211_500network.save_svg('Pt211_500network.svg')

