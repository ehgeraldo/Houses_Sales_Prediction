#==============================
# Imports
#==============================

import pandas as pd
import numpy  as np

import seaborn           as sns
import streamlit         as st
import plotly.express    as px


#==============================

# Site Header
st.title( 'Houses Sales Prediction' )
st.markdown( 'House Rocket Data Analysis' )

# Load data
#uploaded_file = st.file_uploader("Choose a file")
#if uploaded_file is not None:
#  data = pd.read_csv(uploaded_file)
  
st.header('Load dat')
data = pd.read_csv('/home/ehgeraldo/repos/Houses_Sales_Prediction/data/kc_house_data.csv')


# Data Head
st.write(data)

#==============================
# filters
#==============================
# filter bedrooms
bedrooms = st.sidebar.multiselect(
    'Number of Bedrooms', 
    data['bedrooms'].unique() )

#st.write( 'You choose', bedrooms[0] )

df = data[data['bedrooms'].isin(bedrooms)]

# data dimension
st.write( 'Number of Rows:', data.shape[0] )
st.write( 'Number of Cols:', data.shape[1] )

# data types
#st.header( 'Data Types' )
#st.write( data.dtypes )

#==============================
# data descriptive
#==============================

num_attributes = data.select_dtypes( include=['int64','float64'])

# central tendency - media, mediana
ct1 = pd.DataFrame(num_attributes.apply( np.mean ) ).T
ct2 = pd.DataFrame(num_attributes.apply( np.median ) ).T

# dispersion - std, min, max, range, skew, kurtosis
d1 = pd.DataFrame( num_attributes.apply( np.std ) ).T
d2 = pd.DataFrame(num_attributes.apply( min ) ).T
d3 = pd.DataFrame(num_attributes.apply( max ) ).T
d4 = pd.DataFrame( num_attributes.apply( lambda x: x.max() - x.min() ) ).T
d5 = pd.DataFrame( num_attributes.apply( lambda x: x.skew() ) ).T
d6 = pd.DataFrame( num_attributes.apply( lambda x: x.kurtosis() ) ).T

m = pd.concat( [ d2, d3, d4, ct1, ct2, d1, d5, d6] ).T.reset_index()
m.columns = [ 'atributes','min', 'max','range','mean','median','std','skew','kurtosis']

pd.set_option('display.float_format', lambda x: '%.5f' % x)

st.header( 'Data Descriptive' )
st.dataframe( m )

#==============================
# map
#==============================

# define level of prices
for i in range( len( data ) ):
    if data.loc[i, 'price'] <= 321950:
        data.loc[i, 'level'] = 0

    elif ( data.loc[i,'price'] > 321950 ) & ( data.loc[i,'price'] <= 450000 ):
        data.loc[i, 'level'] = 1

    elif ( data.loc[i,'price'] > 450000 ) & ( data.loc[i,'price'] <= 645000 ):
        data.loc[i, 'level'] = 2

    else:
        data.loc[i, 'level'] = 3


# plot map
st.title( 'House Rocket Map' )
is_check = st.checkbox( 'Display Map')

# filters
price_min = int( data['price'].min() )
price_max = int( data['price'].max() )
price_avg = int( data['price'].median() )
price_slider = st.slider('Price Range', price_min, price_max, price_avg)

if is_check:
    # select rows
    houses = data[data['price'] < price_slider][['id','lat','long',
                                                 'price', 'level']]

    # draw map
    fig = px.scatter_mapbox( 
        houses, 
        lat="lat", 
        lon="long", 
        color="level", 
        size="price",
        color_continuous_scale=px.colors.cyclical.IceFire, 
        size_max=15, 
        zoom=10 )

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(height=600, margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart( fig )

