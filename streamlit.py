#==============================
# Imports
#==============================

import pandas as pd
import numpy  as np

import seaborn           as sns
import streamlit         as st
import plotly.express    as px


#==============================

col1, col2, col3 = st.columns([1,2,1])
col1.markdown('👏 Welcome to my app!!!')
col3.markdown('Info about dataset')
col3.markdown('https://bityli.com/jGlQ9')
col3.markdown('https://bityli.com/jGlQ9')

#==============================
# Wide view
#==============================

#st.set_page_config( layout='wide' )

# Site Header
st.title( 'Houses Sales Prediction' )
st.markdown( 'House Rocket Data Analysis' )

# Load data
#uploaded_file = st.file_uploader("Choose a file")
#if uploaded_file is not None:
#  data = pd.read_csv(uploaded_file)
  
st.header('Load data')
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

# add new features
data['price_m2'] = data['price'] / data['sqft_lot'] 

# =======================
# Data Overview
# =======================
f_attributes = st.sidebar.multiselect( 'Enter columns', data.columns ) 
f_zipcode = st.sidebar.multiselect( 
    'Enter zipcode', 
    data['zipcode'].unique() )

st.title( 'Data Overview' )
# data dimension

if ( f_zipcode != [] ) & ( f_attributes != [] ):
    data = data.loc[data['zipcode'].isin( f_zipcode ), f_attributes]

elif ( f_zipcode != [] ) & ( f_attributes == [] ):
    data = data.loc[data['zipcode'].isin( f_zipcode ), :]

elif ( f_zipcode == [] ) & ( f_attributes != [] ):
    data = data.loc[:, f_attributes]

else:
    data = data.copy()

st.dataframe( data )

#c1, c2 = st.columns((1, 1) )  

# Average metrics
df1 = data[['id', 'zipcode']].groupby( 'zipcode' ).count().reset_index()
df2 = data[['price', 'zipcode']].groupby( 'zipcode').mean().reset_index()
df3 = data[['sqft_living', 'zipcode']].groupby( 'zipcode').mean().reset_index()
df4 = data[['price_m2', 'zipcode']].groupby( 'zipcode').mean().reset_index()


# merge
m1 = pd.merge( df1, df2, on='zipcode', how='inner' )
m2 = pd.merge( m1, df3, on='zipcode', how='inner' )
df = pd.merge( m2, df4, on='zipcode', how='inner' )

df.columns = ['ZIPCODE', 'TOTAL HOUSES', 'PRICE', 'SQRT LIVING',
              'PRICE/M2']

#c1.header( 'Average Values' )
#c1.dataframe( df, height=600 )

st.header( 'Average Values' )
st.dataframe( df, height=600 )

# Statistic Descriptive
num_attributes = data.select_dtypes( include=['int64', 'float64'] )
media =   pd.DataFrame( num_attributes.apply( np.mean ) )
mediana = pd.DataFrame( num_attributes.apply( np.median ) )
std =     pd.DataFrame( num_attributes.apply( np.std ) )

max_ = pd.DataFrame( num_attributes.apply( np.max ) ) 
min_ = pd.DataFrame( num_attributes.apply( np.min ) ) 

df1 = pd.concat([max_, min_, media, mediana, std], axis=1 ).reset_index()

df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std'] 

#c2.header( 'Descriptive Analysis' )
#c2.dataframe( df1, height=800 )

st.header( 'Descriptive Analysis' )
st.dataframe( df1, height=800 )




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

