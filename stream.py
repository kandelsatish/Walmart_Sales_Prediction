import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore")


st.write("""
# Walmart Sales Prediction App

This app predicts the **Weekly Sales of Retail giant Walmart**!
""")
st.write('---')


# Set a background image URL
background_image = 'https://raw.githubusercontent.com/kandelsatish/Walmart_Sales_Prediction/main/static/images/drk.png'
html_code  = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url('{background_image}');
background-size: 110%;
background-position: top left;
background-repeat:repeat;
background-attachment: local;
}}
</style>
"""

# Display the HTML code using st.markdown
st.markdown(html_code, unsafe_allow_html=True)


#load the trained ML model
random_forest_reg_model = pickle.load(open(r'model/random_forest_reg_model.pkl','rb'))
elasticnet_model = pickle.load(open(r'model/elasticnet_model.pkl','rb'))
Lasso_regression_model = pickle.load(open(r'model/Lasso_regression_model.pkl','rb'))
Ridge_regression_model = pickle.load(open(r'model/Ridge_regression_model.pkl','rb'))
Linear_regression_model = pickle.load(open(r'model/Linear_regression_model.pkl','rb'))



# Sidebar
# Header of Specify Input Parameters
st.header('Specify Input Parameters')
def user_input_features():
    Temperature = st.text_input("Temperature")
    Fuel_Price = st.text_input("Fuel_Price")
    Unemployment = st.text_input("Unemployment")
    CPI = st.text_input("CPI")
    Store = st.slider('Store',1 , 45)


    data = {'Temperature': Temperature,
            'Fuel_Price': Fuel_Price,
            'Unemployment': Unemployment,
            'CPI': CPI,
            'Store': Store}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Prediction Models:',
                          
                          ['Select a model',
                           'Random Forest Regressor',
                           'ElastiNet Model',
                           'Ridge Regression Model',
                           'Lasso Regression Model',
                           'Linear Regression Model'],
                          default_index=0)
# Main Panel
# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Combine all features into a single DataFrame
input_data = df.drop('Store', axis=1)
store_dummies = pd.get_dummies(pd.Series(df['Store']), prefix='Store')
store_dummies = store_dummies.reindex(columns=['Store_' + str(i) for i in range(1, 46)], fill_value=0)
input_data = pd.concat([store_dummies, input_data], axis=1)

# Make prediction using the model:
st.header('Prediction of Weekly sales of Walmart: ')
if (selected == 'Random Forest Regressor'):
    model = random_forest_reg_model
    
elif(selected == 'ElastiNet Model'):
    model = elasticnet_model
    
elif(selected == 'Ridge Regression Model'):
    model = Ridge_regression_model
    
elif(selected == 'Lasso Regression Model'):
    model = Lasso_regression_model
    
elif(selected == 'Linear Regression Model'):
    model = Linear_regression_model
    

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.success('The estimated weekly sales for the upcoming week is {} US Dollar'.format(prediction[0]))

st.write('---')



