# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 13:34:21 2020

@author: mathew.a.pazhur
"""

#Importing  -------------------------
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from nltk.stem import WordNetLemmatizer
import re  
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import mysql.connector


#Importing stop words
stop_words = set(stopwords.words('english'))

#Mysql connection
mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="recom_eng_test"
)

mycursor = mydb.cursor()

#Total Products

mycursor = mydb.cursor()

mycursor.execute("SELECT * FROM product_test")

myresult = mycursor.fetchall()

products=pd.DataFrame(myresult, columns=['product_id','product_name','product_desc','p_type','temp'])


#Enter Customer history in cust_history dataframe

customer_id=int(input("Enter customer ID : "))

mycursor.execute("""SELECT o.order_id,o.customer_id,o.store_id,o.product_id,p.product_name,p.product_description,p.p_type,p.temp
from orders_test_2 o join customer c on o.customer_id=c.customer_id JOIN
product_test p on o.product_id=p.product_id JOIN
store_test_2 s on o.store_id = s.store_id and o.product_id=s.product_id
where o.customer_id=%s
order by 3,2,4""", (customer_id,))

myresult = mycursor.fetchall()

cust_history=pd.DataFrame(myresult, columns=['order_id','customer_id','store_id','product_id','product_name','product_desc','p_type','temp'])

#Enter Store menu in products dataframe

store_id=int(input("Enter Store ID where customer is visiting : "))

mycursor.execute("""SELECT s.store_id,s.store_name,s.product_id,p.product_name 
from store_test_2 s join product_test p on s.product_id=p.product_id
where store_id=%s""",(store_id,))

myresult = mycursor.fetchall()

store_products=pd.DataFrame(myresult, columns=['store_id','store_name','product_id','product_name'])




#-------------------------------------------dividing cust_history
c_h_burger=cust_history.loc[(cust_history['p_type']=='burger')|(cust_history['p_type']=='roll')]
c_h_drink=cust_history.loc[(cust_history['p_type']=='beverage')| (cust_history['p_type']=='coffee')]
c_h_dessert=cust_history.loc[(cust_history['p_type']=='dessert')]
c_h_sides=cust_history.loc[(cust_history['p_type']=='sides')|(cust_history['p_type']=='bfast')]

#------------------------------------------dividing products
tot_burger=products.loc[(products['p_type']=='burger')|(products['p_type']=='roll')]
tot_drink=products.loc[(products['p_type']=='beverage')| (products['p_type']=='coffee')]
tot_dessert=products.loc[(products['p_type']=='dessert')]
tot_sides=products.loc[(products['p_type']=='sides')|(products['p_type']=='bfast')]

#Finding top products of customer

def top_prod(hist_df):               #function finds top 1 product
    cust_top_prod_df=hist_df
    
    a=cust_top_prod_df.groupby(['product_id']).size()                   #group by size
    cust_top_prod_df=cust_top_prod_df.drop(columns=['order_id'])        #drop order id column for dropping duplicates
    a=a.sort_values(ascending=False)                                    #sort by size
    a=a[:1]                                                             #top file id
                                                      
    l=list(a.index)                                                     #convert series to list
    cust_top_prod_df=cust_top_prod_df[cust_top_prod_df['product_id'].isin(l)] #take top product
    
    cust_top_prod_df=cust_top_prod_df.drop_duplicates()                 #remove suplicates
    return(cust_top_prod_df)


cust_top_prod_df=top_prod(cust_history)                                 #top product


cust_top_burger_df=top_prod(c_h_burger)                                 #top burger
cust_top_drink_df=top_prod(c_h_drink)      
cust_top_dessert_df=top_prod(c_h_dessert)                               #top dessert
cust_top_sides_df=top_prod(c_h_sides)                                   #top sides




# cust_top_prod_df=cust_top_prod_df.drop(columns=['customer_id','store_id','product_desc'])
# cust_top_prod_df.drop_duplicates(keep='first',inplace=True)
#Preprocessing -----------------------

X=products['product_desc']

X = X.reset_index(drop=True)

documents = []
stemmer = WordNetLemmatizer()
doc_cleaned_string=''


for sen in range(0, len(X)):  
    # Remove all the special characters
    document = re.sub(r'[^a-zA-Z0-9-]', ' ', str(X[sen]))
    
    #Remove Whitespaces
    #document = document.strip()

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    #Tokenizing sentence
    doc_word_tokens = word_tokenize(document)
    
    #Removing stopwords
    doc_cleaned_list = [w for w in doc_word_tokens if not w in stop_words] 
    
    #Creating cleaned string from cleaned list
    for x in doc_cleaned_list:
        doc_cleaned_string=doc_cleaned_string+x+' '
     
    #Adding string to list
    documents.append(doc_cleaned_string)
    
    doc_cleaned_string=''
    
      
# #Creating Tfidf vector object and removing stop words
tfidf = TfidfVectorizer(stop_words='english')

# #Replacing null values with empty string
products['product_desc']=products['product_desc'].fillna('')

# #Creating tfidf matrix
tfidf_matrix = tfidf.fit_transform(documents)

#Use Cosine formula to get similarity score
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#give indices to products to identify them
indices = pd.Series(products.index, index=products['product_id']).drop_duplicates()

products['preprocessed_desc']=documents

fin_recomm=pd.DataFrame(columns=('product_id','product_name'))
intermed_list=[]

#recommendation function -----------------

def recomm(prod_id, fl, cosine_sim=cosine_sim):
    
    idx = indices[prod_id]
    
    # Get the pairwsie similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    #Store products of each type
    
    store_burger=store_products[store_products['product_id'].isin(list(tot_burger['product_id']))]
    store_dessert=store_products[store_products['product_id'].isin(list(tot_dessert['product_id']))]
    store_drink=store_products[store_products['product_id'].isin(list(tot_drink['product_id']))]
    store_sides=store_products[store_products['product_id'].isin(list(tot_sides['product_id']))]

    if(fl==1):
        print("-------------------------")
        print(store_drink)
        print("-------------------------")
        
    # print(store_burger)
    # print("-------------------------")
    # print(store_dessert)
    # print("-------------------------")

    # print(store_sides)
    # Get the products indices
    product_indices = [i[0] for i in sim_scores]
    pi2=[]
    
    if(fl==1):
        for alpha in range(len(product_indices)):
            for beta in store_burger.product_id:
                if(product_indices[alpha]+1==beta):
                    pi2.append(product_indices[alpha])
    elif(fl==2):
        for alpha in range(len(product_indices)):
            for beta in store_drink.product_id:
                if(product_indices[alpha]+1==beta):
                    pi2.append(product_indices[alpha])
    elif(fl==3):
        for alpha in range(len(product_indices)):
            for beta in store_sides.product_id:
                if(product_indices[alpha]+1==beta):
                    pi2.append(product_indices[alpha])
            
    elif(fl==4):
        for alpha in range(len(product_indices)):
            for beta in store_dessert.product_id:
                if(product_indices[alpha]+1==beta):
                    pi2.append(product_indices[alpha])
    
    

    pi2=pi2[:10]
    
    #Testing performance
    
    # simtest=[]
    # for x in sim_scores:
    #     if(int(x[0]) in pi2):
    #         simtest.append(x[0:2])
    
    # Return the top 3 most similar products
    
    out=products.iloc[pi2, [0,1]]
    out=out[:3]                                   #top 3 recommendations
    
    return(out)
        
#Code for recommendations of all products

print("For Burgers : ")
burg_recomm=recomm(int(cust_top_burger_df.product_id),1)
print(burg_recomm)

print("For Drinks: ")
drink_recomm=recomm(int(cust_top_drink_df.product_id),2)
print(drink_recomm)

print("For sides and breakfast : ")
sides_recomm=recomm(int(cust_top_sides_df.product_id),3)
print(sides_recomm)
      
print("For desserts : ")
dessert_recomm=recomm(int(cust_top_dessert_df.product_id),4)
print(dessert_recomm)
      








