#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Installing packages at notebook level
dbutils.library.installPyPI("nltk")
dbutils.library.installPyPI("pytest")
dbutils.library.installPyPI("gensim")
dbutils.library.installPyPI("bs4")
dbutils.library.installPyPI("scipy", "1.2.1")
dbutils.library.installPyPI("scikit-learn", "0.23.2")
dbutils.library.installPyPI("imbalanced-learn")
dbutils.library.installPyPI('xgboost' , "1.0.2") 
dbutils.library.installPyPI('lightgbm', "2.3.1") 
dbutils.library.installPyPI('catboost') 
dbutils.library.restartPython()


# In[ ]:


get_ipython().run_line_magic('run', '/Shared/modelling/Data_Science_Pipeline/DS_pipeline_functions #running functions from a different notebook')


# In[ ]:


#importing packages
import pandas as pd
import numpy as np
import nltk
nltk.data.path.append('/dbfs/FileStore/')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import pyspark.sql.functions as F
from pyspark.sql.window import Window as W
#from pyspark.sql.functions import *
from pyspark.sql.functions import when, lit, col
from pyspark.sql.functions import countDistinct
from pyspark.sql.types import IntegerType
from pyspark.sql.functions import regexp_extract


# In[ ]:


get_ipython().run_line_magic('sql', '')
REFRESH TABLE aca_pre_prod.prm_claims


# In[ ]:


prm_claims = spark.read.parquet("dbfs:/mnt/prod/data/prod/prm/claims/")
#prm_claims = spark.table('aca_pre_prod.prm_claims')

#prm_payments = spark.read.parquet("dbfs:/mnt/prod/data/prod/prm/payments/")
prm_payments = spark.table('aca_prod.prm_payments')
prm_claimant = spark.table('aca_prod.prm_claimants_motor') 
gw_exposure = spark.table('aca_prod.int_guidewire_claim_exposure')


# In[ ]:


load_date = dbutils.fs.ls("/mnt/prod/raw/visualfiles/data/aca_extract/vf_tpfd/")[-1].name
print ("check that the date makes sense (example: if working on data for oct data might be 20201110 october load date): " , load_date)
vf_tpfd = spark.read.option("header","true").csv('/mnt/prod/raw/visualfiles/data/aca_extract/vf_tpfd/{}/TPFD.csv'.format(load_date)) # r script

#int_pii=spark.read.parquet("dbfs:/mnt/consumableraw/data/acs/data/ACSOWNER.ASA_ACS_LOSS_DETAIL/InternalPII/")
#ext_pii=spark.read.parquet("/mnt/consumableraw/data/acs/data/ACSOWNER.ASA_ACS_LOSS_DETAIL/ExternalPII/")


# In[ ]:


#################Create a claim spine
claim_spine = prm_claims.selectExpr('data_source', 'claim_number', 'lob', 'claim_loss_date', 'claim_notification_date', 
                                    'peril_code', 'peril_description', 'circumstances_of_accident','claim_status', 
                                    'claim_settlement_reason', 'cause_code','cause_description', 'line_of_business_detailed').distinct()

claim_spine.count()


# In[ ]:


########################Join claimant table to spine
# expect higher count because one claim will have multiple claimants. 
#prm_claimants table will give more info about claims
claim_spine = claim_spine.join(prm_claimant.selectExpr('claim_number', 'claimant_number', 'data_source', 'claimant_type', 'loss_type',
                                                       'incident_severity_desc', 'speed_at_impact','is_injured', 'is_vehicle_parked',
                                                       'is_passenger','vehicle_damaged_desc','insured_vehicle_value','incident_severity_desc',
                                                      'can_vehicle_be_driven', 'total_loss_vehicle', 'claimant_type',  
                                                      'vehicle_manufacture_year'), on = ['claim_number', 'data_source'], how = 'left').distinct()
claim_spine.count()


# In[ ]:


#########some EDA
claim_spine.select(F.min('claim_notification_date'),F.max('claim_notification_date')).show()


# In[ ]:


###################Filter for Injury Payments
# filter for actual motor injury payments made, keep 'paid' amounts and latest 'pending' amounts
from pyspark.sql.window import Window as W
W1 = W.partitionBy('claim_number', 'claimant_number')

injury_payments = prm_payments.withColumn('max', F.when(F.col('payment_date') == F.max(F.col('payment_date')).over(W1), 1).otherwise(0))                  .orderBy('claim_number', 'claimant_number', 'payee_name', 'payment_date')                  .filter(F.col('section') == 'TPI')                  .filter(F.col('clean_payment_status') == 'Paid')                   .filter((F.col("transaction_type") == "1") | (F.col("transaction_type").isNull()))                   .distinct()


# In[ ]:


#########################Create Payment features at Claimant level
# Create payment feature column to identify the various types of payments involved for each claim based on the payment_desc feature
#Payments for a claim may involve litigation, medical bills, hospital charges, general damages etc
payment_feat = injury_payments                  .withColumn('ft_payment_desc_category', F.when(F.col('payment_grouping1')=='Our Solicitors','ft_case_pay_our_legal_spend')                                                    .when(F.col('payment_grouping1')=='General Damages','ft_case_pay_psla_amount')                                                    .when(F.col('payment_grouping1')=='Claimant Solicitors','ft_case_pay_claimant_legal_spend')                                                    .when(((F.col('payment_desc')=='TPI - Medical Expenses') | (F.col('payment_desc')=='Medical Reports') | (F.col('payment_desc')=='Medical Report Fee')) &
                                                          (F.col('payment_amount')== 216),'ft_case_pay_claimant_legal_spend')\
                              .when(((F.col('payment_desc')=='TPI - Medical Expenses') | (F.col('payment_desc')=='Medical Reports') | (F.col('payment_desc')=='Medical Report Fee')) & (F.col('payment_amount')==
                              780),'ft_case_pay_claimant_legal_spend')\
                              .when((F.col('payment_desc')=='Claimant Costs, Fees and Disbursements'),'ft_case_pay_claimant_legal_spend')\
                              .when(((F.col('payment_desc')=='TPPI - Hospital Charges') | (F.col('payment_desc')=='TPI - Hospital Charges') | (F.col('payment_desc')=='NHS In Patient') | (F.col('payment_desc')=='NHS Ambulance')
                                     | (F.col('payment_desc')=='NHS Inpatients Charges')),'ft_case_pay_other_costs')\
                              .when((F.col('payment_grouping1')=='Special Damages'),'ft_case_pay_special_damages_spend')\
                              .otherwise('ft_case_pay_other_costs')\
                             )\
.drop_duplicates()


# In[ ]:


#################### count will decrease because only keeping TPI payments
payments_claim_spine = claim_spine.join(payment_feat, on = ['claim_number', 'claimant_number', 'lob', 'data_source'], how = 'right')


# In[ ]:


############################ pivot table to get payment types across
payment_pivot = payments_claim_spine                        .groupby('data_source', 'claim_number', 'claimant_number', 'lob', 'claim_notification_date', 
                                 'claim_status', 'section','clean_payment_status','transaction_type')\
                        .pivot('ft_payment_desc_category')\
                        .agg(F.sum('payment_amount'))\
                        .selectExpr('*', 'coalesce(ft_case_pay_our_legal_spend,0) + coalesce(ft_case_pay_claimant_legal_spend,0) as ft_case_pay_total_legal_spend')\
.selectExpr('*', 'coalesce(ft_case_pay_our_legal_spend,0) + coalesce(ft_case_pay_claimant_legal_spend,0) +  coalesce(ft_case_pay_special_damages_spend,0) + coalesce(ft_case_pay_psla_amount,0) + coalesce(ft_case_pay_other_costs,0) as ft_pay_total_spend')\


# In[ ]:


##########################Create flags for large loss
# Create flags for large losses
payment_pivot = payment_pivot.withColumn("ft_large_loss1", F.when(F.col('ft_pay_total_spend') > 100000, F.lit(1)).otherwise(F.lit(0)))
payment_pivot = payment_pivot.withColumn("ft_large_loss2", F.when(F.col('ft_pay_total_spend') > 300000, F.lit(1)).otherwise(F.lit(0)))

payment_pivot = payment_pivot.withColumn("ft_large_loss3", F.when((F.col('ft_pay_total_spend') > 100000) & (F.col('ft_pay_total_spend') < 300000),
                                                                  F.lit(1)).otherwise(F.lit(0)))
payment_pivot.count()


# In[ ]:


##################Filter for Motor injury claims only
ll_motor_inj = payment_pivot.filter((F.col("lob") == "CMTR") | (F.col("lob") == "PMTR") | (F.col("lob") == "DMTR") | (F.col("lob") == "ECU")).filter(F.col('section') == 'TPI').filter(F.col('clean_payment_status') == 'Paid')                   .filter((F.col("transaction_type") == "1") | (F.col("transaction_type").isNull()))                   .distinct().dropDuplicates(['claim_number'])
ll_motor_inj.count()


# In[ ]:


#################Count number of duplicate rows
#please check to make sure that table definitely only has one row per claim
import pyspark.sql.functions as f
from pyspark.sql import Window

w = Window.partitionBy('claim_number', 'data_source')
dp = ll_motor_inj.select('*', f.count('claim_number').over(w).alias('dupeCount'))    .where('dupeCount > 1')    .drop('dupeCount')
display(dp)


# In[ ]:


###########################MOTOR INJ
#check TPIs fields to be used to create payment features
prm_claims = spark.read.parquet("dbfs:/mnt/prod/data/prod/prm/claims/")
#prm_claims = spark.table('aca_prod.prm_claims')
prm_payments = spark.table('aca_prod.prm_payments')
prm_claimant = spark.table('aca_prod.prm_claimants_motor') 

tpi_payments = prm_payments.filter(F.col("lob").isin(["CMTR", "PMTR", "DMTR", "ECU"]))      .filter(F.col('section') == 'TPI').filter(F.col("clean_payment_status") == "Paid")
#display(tpi)


# In[ ]:


#Create motor spine 
import pyspark.sql.functions as F
# read in all claims and filter to select closed claims for required lob
# join to payments on same lob, grouped to sum all payments by claim number
motor_claim_spine = prm_claims.filter(F.col("lob").isin(["CMTR", "PMTR", "DMTR", "ECU"])).filter(F.col("claim_status") == "CLOSED").join(tpi_payments.groupBy("claim_number").agg({"payment_amount":"sum"}),
     on = "claim_number", how = "left")\
.withColumnRenamed("sum(payment_amount)","total_paid")\
.na.fill(value=0,subset=["total_paid"])

motor_claim_spine.count()


# In[ ]:


#####Count number of duplicate rows
import pyspark.sql.functions as f
from pyspark.sql import Window

w = Window.partitionBy('claim_number', 'data_source')
dp = motor_claim_spine.select('*', f.count('claim_number').over(w).alias('dupeCount'))    .where('dupeCount > 1')    .drop('dupeCount')
display(dp)


############### Create flags for large losses
motor_inj_spine = motor_claim_spine.withColumn("ft_large_loss1", F.when(F.col('total_paid') > 100000, F.lit(1)).otherwise(F.lit(0)))
motor_inj_spine = motor_inj_spine.withColumn("ft_large_loss2", F.when(F.col('total_paid') > 300000, F.lit(1)).otherwise(F.lit(0)))

motor_inj_spine = motor_inj_spine.withColumn("ft_large_loss3", F.when((F.col('total_paid') > 100000) & (F.col('total_paid') < 300000),
                                                                  F.lit(1)).otherwise(F.lit(0)))


# In[ ]:


####### expect higher count because one claim will have multiple claimants. 
#prm_claimants table will give more info about claims
motor_inj_spine = motor_inj_spine.join(prm_claimant                                     .filter(F.col("claimant_type")=="THIRD PARTY")                                     .selectExpr('claim_number', 'claimant_number', 'data_source', 'claimant_type', 'loss_type',
                                                       'incident_severity_desc', 'speed_at_impact','is_injured', 'is_vehicle_parked',
                                                       'is_passenger','vehicle_damaged_desc','insured_vehicle_value','incident_severity_desc',
                                                      'can_vehicle_be_driven', 'total_loss_vehicle', 'claimant_type', 
                                                      'vehicle_manufacture_year'), on = ['claim_number', 'data_source'], how ='left').distinct()
#'data_source'


# In[ ]:


#############################check duplicates
from pyspark.sql.functions import desc
dis = motor_inj_spine.groupBy("is_injured").count().orderBy(desc("count"))
display(dis)

###################
motor_inj_mst = motor_inj_spine.dropDuplicates(['claim_number'])
motor_inj_mst.count()


# In[ ]:


###########################
from pyspark.sql.functions import when   
motor_inj_mst1 = motor_inj_mst  .withColumn('is_injured', when(motor_inj_mst['is_injured']=='true', 1).otherwise(0))  .withColumn('is_vehicle_parked', when(motor_inj_mst['is_vehicle_parked']=='true', 1).otherwise(0))  .withColumn('can_vehicle_be_driven', when(motor_inj_mst['can_vehicle_be_driven']=='Yes', 1).otherwise(0))

cols = ['is_injured',
'is_vehicle_parked',
'can_vehicle_be_driven',
'total_loss_vehicle']

motor_inj_mst1 = motor_inj_mst1.na.fill('0', subset=['total_loss_vehicle'])



######################################Convert to pandas dataframe & select ll cases
import pandas as pd
#motor_inj_mst_pd = motor_inj_mst1.toPandas()
print (motor_inj_mst1.count())
motor_inj_mst2 = motor_inj_mst1.filter(motor_inj_mst1['claim_notification_date'] > "2014-01-01")

motor_inj_mst_pd  =  motor_inj_mst2.selectExpr('claim_number', 
                                          'is_injured',
                                          'is_vehicle_parked',
                                          'can_vehicle_be_driven',
                                          'total_loss_vehicle',
                                          'ft_large_loss1',
                                          'total_excess',
                                          'speed_at_impact',
                                          'insured_liability_position_value',
                                          'ft_large_loss2').toPandas()

motor_inj_mst_pd[['is_injured', 'speed_at_impact', 'is_vehicle_parked','can_vehicle_be_driven','total_loss_vehicle','ft_large_loss1',
                'ft_large_loss2', 'total_excess', 'insured_liability_position_value']].apply(pd.to_numeric, errors='coerce').fillna(0)

print(motor_inj_mst_pd.shape)
ll_cases1 = motor_inj_mst_pd[motor_inj_mst_pd["ft_large_loss1"] == 1]
ll_cases2 = motor_inj_mst_pd[motor_inj_mst_pd["ft_large_loss2"] == 1]


other_cases1 = motor_inj_mst_pd[motor_inj_mst_pd["ft_large_loss1"] == 0]
other_cases2 = motor_inj_mst_pd[motor_inj_mst_pd["ft_large_loss2"] == 0]


#####################################
#ll_cases1 = motor_inj_mst_pd[motor_inj_mst_pd["ft_large_loss1"] == 1]
ll_cases1.shape
motor_inj_mst_pd.sample()
motor_inj_mst_pd.groupby(["ft_large_loss1"]).size()
motor_inj_mst1.count()
print(len(ll_cases1.index))
print(len(ll_cases2.index))


# In[ ]:


#############################Notes Analyses
#Select only notes from table spine
df_notes = motor_inj_mst1.select('claim_number','circumstances_of_accident', 'transaction_date', 'cause_description').filter(F.col('circumstances_of_accident').isNotNull())

#notes_new = motor_inj_mst1.select('claim_number','circumstances_of_accident').filter(F.col('circumstances_of_accident').isNotNull())

df_notes_new =  motor_inj_mst2.select('claim_number', 'circumstances_of_accident', 'is_injured', 'is_vehicle_parked','can_vehicle_be_driven',
                                          'total_loss_vehicle', 'speed_at_impact', 'total_excess','insured_liability_position_value',
                                          'ft_large_loss1', 'ft_large_loss2').filter(F.col('circumstances_of_accident').isNotNull())


#filter(F.col('cause_description')


# In[ ]:


###################################################
def drop_null_columns(df, limit_prop):
    """
    This function drops all columns which contain null values.
    Limit_prop should be between 0-1, indicates what max prop of null values is
    :param df: A PySpark DataFrame
    """
    df = df.cache()
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).cache()
    null_counts = null_counts.collect()[0].asDict()
    limit = df.count() * limit_prop
    to_drop = [k for k, v in null_counts.items() if v >= limit]
    df = df.drop(*to_drop)

    return df


# In[ ]:


############################# drop columns with >99% null values
master = drop_null_columns(motor_inj_mst1, 0.99)

notes = master.select('claim_number', 'transaction_date', 'circumstances_of_accident').filter(F.col('circumstances_of_accident').isNotNull())

#rename columns
notes = notes.withColumn("fnol_notes", F.col('circumstances_of_accident'))
notes = notes.select(['claim_number','transaction_date','fnol_notes'])


# In[ ]:


#####################FNOL NOTES Analyses
# convert datetimes to dates
from pyspark.sql.types import DateType, TimestampType
from pyspark.sql.functions import from_unixtime, unix_timestamp
notes = notes.withColumn("notes_date",
                         from_unixtime(unix_timestamp(notes["transaction_date"],
                                                      format="yyyy-MM-dd HH:mm:ss"),
                                       "yyyy-MM-dd").cast(DateType()))\
.withColumn("notes_date_time",from_unixtime(unix_timestamp(notes["transaction_date"],
                                                      format="yyyy-MM-dd HH:mm:ss"),
           "yyyy-MM-dd HH:mm:ss").cast(TimestampType()))

# join master to notes - only keep notes made on date of claim
fnol_notes = master[["claim_number", "claim_notification_date"]].join(notes, (master["claim_number"] == notes["claim_number"]) & (master["claim_notification_date"] == notes["notes_date"]), how = "inner").drop(master["claim_number"]).orderBy(["claim_number", "notes_date_time"])

# remove initial utterance and blackboard
fnol_notes = fnol_notes.where(~(fnol_notes["fnol_notes"].like("Blackboard%")) &                       ~(fnol_notes["fnol_notes"].like("Initial Utterance%")))

# concatenate all notes taken on claim notification date
fnol_notes = fnol_notes.groupBy(["claim_number", "notes_date"]).agg(F.count("claim_number").alias("number_of_notes"),
     F.concat_ws("~",F.collect_list("fnol_notes")).alias("fnol_notes"))

# convert all notes to lower case
fnol_notes = fnol_notes.withColumn("fnol_notes", F.lower(F.col("fnol_notes")))


# In[ ]:


# #################Add column for notes across all dates
# join master to notes - keep notes from all dates on claim
all_notes = master[["claim_number"]].join(notes, (master["claim_number"] == notes["claim_number"]), how = "inner").drop(master["claim_number"]).orderBy(["claim_number", "notes_date_time"])

# remove initial utterance and blackboard
all_notes = all_notes.where(~(all_notes['fnol_notes'].like("Blackboard%")) &                       ~(all_notes['fnol_notes'].like("Initial Utterance%")))

# concatenate all notes 
all_notes = all_notes.groupBy(["claim_number"]).agg(F.count("claim_number").alias("number_of_notes"),
     F.concat_ws("~",F.collect_list('fnol_notes')).alias('fnol_notes'))

# convert all notes to lower case
all_notes = all_notes.withColumn('fnol_notes', F.lower(F.col('fnol_notes')))

#rename columns
all_notes = all_notes.withColumn('all_notes', F.col('fnol_notes'))
all_notes = all_notes.withColumn("number_of_notes_all", F.col("number_of_notes"))
all_notes = all_notes.select(['claim_number','number_of_notes_all', 'all_notes'])

#display(all_notes)
#Join notes columns together
#final_notes = notes.join(all_notes, (notes['claim_number'] == all_notes['claim_number']), how='inner').drop(all_notes['claim_number'])



####### drop columns with >99% null values
notes_new = drop_null_columns(df_notes_new, 0.99)


# In[ ]:


import re
import string
import logging
from typing import Any, Dict
import pyspark.sql
from pyspark.sql import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import substring, regexp_replace, col, isnan, when, count, udf, lit
from pyspark.sql.types import *
from pyspark.ml.feature import *
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
nltk.data.path.append('/dbfs/FileStore/')
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[ ]:


############################### fill missing value in notes
# remove non-alphanumeric characters (apart from .) 
# remove multiple spaces
# convert to lower case

#Join other features table to table with notes table
#notes_new = motor_inj_mst2.join(notes_new, on = ['claim_number'], how = 'left').distinct()

notes1 = df_notes_new.toPandas() 
notes1["circumstances_of_accident"] = notes1["circumstances_of_accident"].fillna("no note").str.replace("[^a-zA-Z0-9.]"," ").str.replace(' +', ' ').str.lower()

#notes1["cause_description"] = notes1["cause_description"]\
#.fillna("no note").str.replace("[^a-zA-Z0-9.]"," ").str.replace(' +', ' ').str.lower()


# In[ ]:


############################COUNT MOST FREQUENTLY OCCURING WORDS
# define the words count function
def wordCount(wordListDF):
    """Creates a DataFrame with word counts.
    Args:
        wordListDF (DataFrame of str): A DataFrame consisting of one string column called 'word'.
    Returns:
        DataFrame of (str, int): A DataFrame containing 'word' and 'count' columns.
    """
    return (wordListDF
                .groupBy('word').count())


# In[ ]:


##############################Identify stop words to be removed
#https://databricks-prod-#cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3328674740105987/4033840715400609/6441317451288404/latest.html
stop_words = ['policy', 'holder', 'insured', 'the','them','ph', 'tp', 'tpy', 'insd', 'insured', 'ins', 'caused', 
              'causing','whilst', 'third', 'party', 'claim', 'onto', 'due', 'failed', 'customer', 'client',
              'clients', 'insureds', 'working','customers','contractor','claimant','cause','recoveries',
              'unknown','and','also','advised','call','called','damage','email','sent','received','contact','back',
             'diary','update','bvs','next','payment','note','crawford','vat','mr','excess','mrs','miss','ms','dr',
             'sedgwick','cunningham','lindsay','xactware','oriel','payments','eft','axa']


# In[ ]:


#################################removing stopwords
stop_words.extend(StopWordsRemover().getStopWords())
stop_words = list(set(stop_words))#optionnal
remover = StopWordsRemover(inputCol="split", outputCol="filtered", stopWords=stop_words


# In[ ]:


#######################splitting notes/sentences into words
notes_split = df_notes.select(F.split(F.col('circumstances_of_accident'),'\s+').alias('split').cast("array<string>"))
#display(notes_split)

#splitting notes/sentences into words
notes_split1 = df_notes.select(F.split(F.col('cause_description'),'\s+').alias('split').cast("array<string>"))
#display(notes_split1)


# In[ ]:


##################################Removing the stop words from created Notes table above
notes_rmv = remover.transform(notes_split)
#display(notes_rmv)

#notes_rmv1 = remover.transform(notes_split1)
#display(notes_rmv1)


######################################Filter wrds out into a single column from circumstances of accident feature
ind_words = notes_rmv.select(F.explode(F.col('filtered')).alias('word')).where(F.col('word')!='')

#Count individual words in Notes and order them by most frequently occuring
word_cnt = wordCount(ind_words)
display(word_cnt.orderBy('count',ascending=False))


#######################Filter words out into a single column
ind_words1 = notes_split1.select(F.explode(F.col('split')).alias('word')).where(F.col('word')!='')

#Count individual words in Notes and order them by most frequently occuring
word_cnt1 = wordCount(ind_words1)
display(word_cnt1.orderBy('count', ascending=False))


# In[ ]:


#############################FEATURE ENGINEERING FOR NOTES
# single token (single word) matches
# Dict key has 3 values: 1. a list of synonyms, 2. a list of negations, 3. the position of the negation with respect to the token
tokenlist = {
  "cyclist":[["cyclist", "bike", "motorcycle", "motorbike", "pedestrians/cyclists"], ["no"], [-1]], 
  "ambulance":[["ambulance", "ambula", "ambulances", "helicopter"], [], []],
  "fatality":[["fatal", "fatality", "death", "die", "demise", "dying", "decease", "deceased", "died", "dies", "dead"], [], []],
  "injury":[["injury", "injuries", "burns", "injured"], ["no"], [-1]],
  "motorway":[["motorways", "motorway"], [], []],
  "shop":[["shop", "shops"], [], []],
  "property":[["property", "properties", "house", "houses", "accommodation"], [], []],
  "child":[["child", "children", "kid", "kids", "baby", "babies", "toddler", "toddlers", "infant", "infants"], [], []]
  }


# In[ ]:


######tokenize function
from nltk import word_tokenize
# function to tokenize text
def tokenize(text):
    return word_tokenize(text)


# In[ ]:


################################Synonym replacement function
# function to replace synonyms
def replace_synonyms(text, tokens):
    allwords = {}
    for v in tokens:
        for w in tokens[v][0]:
            allwords.update({w:v})
    newtext = []
    for j in text:
        if j in allwords:
            newtext.append(allwords[j])
        else:
            newtext.append(j)
    return newtext


# In[ ]:


###########################Token matching function
# function to match tokens
# NB value of 1 is returned for each occurrence of the token alone, -1 is returned for eah occurrence accompanied by a negation, and 0 is returned if no matches are found
def match_tokens(text, tokens):
    words = {}
    for index, element in enumerate(text):
        if element in tokens:
            if index > 0:
                if text[index-1] in tokens[element][1]:
                    words[element].append(-1) if element in words else words.update({element:[-1]})
                else:
                    words[element].append(1) if element in words else words.update({element:[1]})
            else:
                words.update({element:[1]})
    for w in tokens:
        if w not in words:
            words.update({w:[0]}) 
    words = dict(sorted(words.items(), key=lambda item: item[0]))
    return words


# In[ ]:


#####################Combine into one processing function
def process_tokens(text, tokens):
  return match_tokens(replace_synonyms(tokenize(text), tokens), tokens)


# In[ ]:


##########Match tokens in training data
# match tokens
notes1["tokens_matched"] = notes1["circumstances_of_accident"].apply(lambda x: process_tokens(x, tokenlist))
notes1.head()


# In[ ]:


###Create features from token matches
# create columns for all token matches
# NB columns can be created for number of occurrences (ie sum) or max etc
for tok in tokenlist:
    notes1[tok] = notes1["tokens_matched"].apply(lambda x: max(x[tok]))


# In[ ]:


#################################################CREATE DICT FOR STRINGS: MULTIWORD PHRASES AND PARTIAL WORDS TO BE MACTCHED
# String matches for Motor Injury keywords and phrases in notes
# Dict key has 2 values: 1. a list of synonyms, 2. a list of phrases containing the synonym that should be ignored
stringlist = {
  "potential_high_speed":[["fast speed", "sped", "speeding", "high speed", "extreme speed"], ["low speed", "mild speeding", "slow speed"]],
  "potential_emergency":[["explosion", "helicopter emergency service", "hem", "resuscitation", "emergency service", "emergency services", "fire engine", "fire", "police", "attended by fire services", "fire incident", "emergency resuscitation", "cutting out from vehicle", "cut out from vehicle", "cut from vehicle", "air lifted", "loss of life"], []],
   "heavy_collision":[["train collision", "collision with a train", "lorry collision", "collision with a lorry", "heavy vehicle collision", "heavy collision", "commercial vehicle collision", "articulated vehicle", "head on collision", "multiple vehicle", "multiple vehicles"], []],
  "many_claimants":[["claimants"], ["multiple claimants", "many claimants", "multiple claims", "many claims"]],
  "child_injury":[["child injury", "children injury", "kid injured", "kids injured", "child injuries", "injured child", "injured children", "injured kid", "injured kids", "injured baby", "injured babies", "injured toddler", "injured toddlers", "injured infant", "injured infants"], []],
"accident_desc":[["windscreen claim", "thrown out of vehicle", "vehicle rolled over", "vehicle turned over", "hit bonnet", "thrown into air", "phd degree student", "fell from roof", "fall from roof", "fell from height", "fall from height", "onto concrete", "in a bad way"], []],
  "illness_type":[["cancer", "asbestosis", "ptsd", "asbestos related cancers", "complex regional pain syndrome", "chronic regional pain syndrome", "severe post traumatic amnesia", "loss of sight", "brachial plexus"], []], 
 "injury_nature":[["loss of use of upper limb", "loss of use of lower limb", "loss of use of upper limbs", "loss of use of lower limbs", "amputations of upper limbs", "amputations of lower limbs", "amputation of upper limbs", "fractured os calcis"], []],  
 "claimant_profession":[["management board representative", "marketing professional", "accountant", "company secretary", "off work", "doctor", "lawyer", "stock market dealer", "television personality"], []], 
 "claimant_circumstance":[["rehabilitation bio psychosocial yellow", "catastrophising", "black flag", "Low expectation about return to work", "fear avoidance behaviour", "area of high unemployment", "lack of job satisfaction", "no contact with work", "surviving spouse", "surviving children", "surviving child", "illness behaviour", "pain behaviour", "pain intensity behaviour", "pain functional behaviour"], []], 
 "accident_outcome":[["distinct to rehabilitation assessment", "wheelchair bound", "addenbrookes hospital", "phd degree student", "studies affected", "brain scan", "personality change", "balance issues", "dizziness", "blurred vision", "cognitive issues", "amnesia", "tinnitus", "subarachnoid haemorrhage", "bleed on brain", "seizures", "epilepsy", "neurosurgery", "unconscious", "loss of consciousness", "post concussion syndrome", "glasgow coma scale", "gcs", "cauda equina", "chiari malformation", "induced coma", "intensive care", "prognosis is uncertain", "uncertain prognosis"], []],
  "injury_severity":[["intra-articular fracture", "intra-articular fractures", "comminuted fracture", "comminuted fractures", "multiple fractures", "polytrauma" "head injury", "head injuries", "lower back injury", "multiple injuries", "spinal injury", "spinal injuries", "chronic pain", "skull fractures", "skull fracture"], ["mild injury", "mild injuries", "lacerations", "cuts", "bruise", "bruises", "lacerations" "contusions"]],
  "accident_region":[["rep of ireland", "republic of ireland"], []],
  "claim_region":[["usa", "canada"], []],
  "damaged_premises":[["business interuption", "loss of business", "aviation industry", "marine industry", "microelectronic industry", "computer industry", "oil industry", "film industry", "supply of goods to the aviation industry", "supply of goods to the marine industry", "supply of goods to the amicroelectronic industry", "supply of goods to the computer industry", "supply of goods to the oil industry", "supply of goods to the film industry", "supply of services to the aviation industry", "supply of services to the marine industry", "supply of services to the microelectronic industry", "supply of services to the computer industry", "supply of services to the oil industry", "supply of services to the film industry"], []] 
}


# In[ ]:


########String matchng function
import re
# function to match strings
def match_strings(text, strings):
    stringmatches = {}
    for index, element in enumerate(strings):
        for str in strings[element][0]:
            include_start_idx = [m.start() for m in re.finditer(str, text)]              
            for include_idx in include_start_idx:
                string_val = 1
                # loop through strings to be excluded, to see if any overlap with string identified
                for exclude_str in strings[element][1]:
                    exclude_start_idx = [m.start() for m in re.finditer(exclude_str, text)]
                    for exclude_idx in exclude_start_idx:
                        if (include_idx >= exclude_idx) & (include_idx <= exclude_idx + len(exclude_str)):
                          string_val = 0
                # update output
                stringmatches[element].append(string_val) if element in stringmatches else stringmatches.update({element:[string_val]})

    for str in strings:
        if str not in stringmatches:
            stringmatches.update({str:[0]}) 
    stringmatches = dict(sorted(stringmatches.items(), key=lambda item: item[0]))
    return stringmatches


# In[ ]:


####Match strings to training data
# match strings
notes1["string_matches"] = notes1["circumstances_of_accident"].apply(lambda x: match_strings(x, stringlist))


# In[ ]:


######Create features from string matches
# create columns for all string matches
# NB columns can be created for number of occurrences (ie sum) or max etc
for str in stringlist:
  notes1[str] = notes1["string_matches"].apply(lambda x: max(x[str]))
notes1.head()

#df = pd.merge(motor_inj_mst_pd, notes1, on='claim_number')
#df.head()


# In[ ]:


#######Import SMOTE & required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import imblearn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

df = notes1[['insured_liability_position_value',
                        'total_excess',
                        'is_injured', 
                        'is_vehicle_parked',
                        'can_vehicle_be_driven',
                        'total_loss_vehicle', 
                        'speed_at_impact',       
                        'cyclist',
                        'ambulance',
                        'motorway',
                        'shop',
                        'property', 
                        'child', 
                        'potential_high_speed',
                        'potential_emergency', 
                        'heavy_collision', 
                        'many_claimants', 
                        'child_injury', 
                        'injury_severity',
                        "accident_desc", 
                        "illness_type",
                        "claimant_profession",
                        "claimant_circumstance",
                        "accident_outcome",
                        'accident_region',
                        'damaged_premises', 
                        'damaged_premises', 
                        'ft_large_loss1']].apply(pd.to_numeric, errors='coerce').fillna(0)


# In[ ]:


#EDA
# Visualize male/female ratio
#sns.countplot(x=df["is_injured"]).set_title("LL Ratio")

# Visualize the classes distributions
sns.countplot(x=df["ft_large_loss1"]).set_title("LL Count")

# Visualize the classes distributions by gender
#sns.countplot(x="ft_large_loss1", hue="is_injured", data=df).set_title('LLCount by Injury')

#Data cleaning
# Check if there are any null values
#df.isnull().values.any()
# Remove null values
#df = df.dropna()
# Check if there are any null values
#df.isnull().values.any()


# In[ ]:


#######################Import the SMOTE-NC
from imblearn.over_sampling import SMOTENC
#Create the oversampler. For SMOTE-NC we need to pinpoint the column position where is the categorical features are. In this case, 'IsActiveMember' is positioned in the second column we input [1] as the parameter. If you have more than one categorical columns, just input all the columns position

##Random undersampling to balance dataset
# Specify features columns
X = df.drop(columns='ft_large_loss1', axis=0)

# Specify target column
y = df['ft_large_loss1']

#smotenc = SMOTENC(categorical_features=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], random_state= 101, sampling_strategy=.2)
#X_sample, y_sample = smotenc.fit_resample(X.values, y.ravel())

smote = SMOTE(random_state = 11)
X_sample, y_sample = smote.fit_resample(X.values, y.ravel())

# Visualize new classes distributions
sns.countplot(y_sample).set_title('Balanced Data Set')


# In[ ]:


#######Model building and performance evaluation
# Import required libraries for performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}


# In[ ]:


# Import required libraries for machine learning classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# Instantiate the machine learning classifiers
log_model = LogisticRegression(max_iter=10000)
svc_model = LinearSVC(dual=False)
dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
gnb_model = GaussianNB()
xgb_model = XGBClassifier()
gbm_model = GradientBoostingClassifier()
hgbm_model = HistGradientBoostingClassifier()
lightgbm_model = LGBMClassifier()


# In[ ]:


# Define the models evaluation function
def models_evaluation(X, y, folds):
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    
    # Perform cross-validation to each machine learning classifier
    log = cross_validate(log_model, X, y, cv=folds, scoring=scoring)
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scoring)
    dtr = cross_validate(dtr_model, X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
    xgb = cross_validate(xgb_model, X, y, cv=folds, scoring=scoring)
    gbm = cross_validate(gbm_model, X, y, cv=folds, scoring=scoring)
    hgbm = cross_validate(hgbm_model, X, y, cv=folds, scoring=scoring)
    lightgbm = cross_validate(lightgbm_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'Logistic Regression':[log['test_accuracy'].mean(),
                                                               log['test_precision'].mean(),
                                                               log['test_recall'].mean(),
                                                               log['test_f1_score'].mean()],
                                       
                                      'Support Vector Classifier':[svc['test_accuracy'].mean(),
                                                                   svc['test_precision'].mean(),
                                                                   svc['test_recall'].mean(),
                                                                   svc['test_f1_score'].mean()],
                                       
                                      'Decision Tree':[dtr['test_accuracy'].mean(),
                                                       dtr['test_precision'].mean(),
                                                       dtr['test_recall'].mean(),
                                                       dtr['test_f1_score'].mean()],
                                       
                                      'Random Forest':[rfc['test_accuracy'].mean(),
                                                       rfc['test_precision'].mean(),
                                                       rfc['test_recall'].mean(),
                                                       rfc['test_f1_score'].mean()],
                                       
                                      'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
                                                              gnb['test_precision'].mean(),
                                                              gnb['test_recall'].mean(),
                                                              gnb['test_f1_score'].mean()],
                                       
                                       'XGBoost':[xgb['test_accuracy'].mean(),
                                                  xgb['test_precision'].mean(),
                                                  xgb['test_recall'].mean(),
                                                  xgb['test_f1_score'].mean()],
                                       
                                       'GradientBoostingClassifier':[gbm['test_accuracy'].mean(),
                                                                     gbm['test_precision'].mean(),
                                                                     gbm['test_recall'].mean(),
                                                                     gbm['test_f1_score'].mean()],
                                       
                                       'Hist GradientBoostingClassifier':[hgbm['test_accuracy'].mean(),
                                                                          hgbm['test_precision'].mean(),
                                                                          hgbm['test_recall'].mean(),
                                                                          hgbm['test_f1_score'].mean()],
                                       
                                       'LGBMClassifier':[lightgbm['test_accuracy'].mean(),
                                                         lightgbm['test_precision'].mean(),
                                                         lightgbm['test_recall'].mean(),
                                                         lightgbm['test_f1_score'].mean()]},
                                       
                                      index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    
    # Return models performance metrics scores data frame
    return(models_scores_table)
  
# Run models_evaluation function
models_evaluation(X_sample, y_sample, 5)


# In[ ]:


#######################################smotenc = SMOTENC([1], random_state = 101)
smotenc = SMOTENC(categorical_features=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], random_state= 101, sampling_strategy=.2)
X_sample1, y_sample1 = smotenc.fit_resample(X.values, y.ravel())

#from imblearn.combine import SMOTEENN
#smt = SMOTEENN(random_state=42)
#X_sample, y_sample = smt.fit_resample(X.values, y.ravel())

# Visualize new classes distributions
sns.countplot(y_sample1).set_title('Balanced Data Set')


# In[ ]:


#######Model building and performance evaluation
# Import required libraries for performance metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

# Define dictionary with performance metrics
scoring = {'accuracy':make_scorer(accuracy_score), 
           'precision':make_scorer(precision_score),
           'recall':make_scorer(recall_score), 
           'f1_score':make_scorer(f1_score)}


# In[ ]:


# Import required libraries for machine learning classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB


# In[ ]:


# Instantiate the machine learning classifiers
log_model = LogisticRegression(max_iter=10000)
svc_model = LinearSVC(dual=False)
dtr_model = DecisionTreeClassifier()
rfc_model = RandomForestClassifier()
gnb_model = GaussianNB()
xgb_model = XGBClassifier()
gbm_model = GradientBoostingClassifier()
hgbm_model = HistGradientBoostingClassifier()
lightgbm_model = LGBMClassifier()


# In[ ]:


# Define the models evaluation function
def models_evaluation(X, y, folds):
    
    '''
    X : data set features
    y : data set target
    folds : number of cross-validation folds
    
    '''
    
    # Perform cross-validation to each machine learning classifier
    log = cross_validate(log_model, X, y, cv=folds, scoring=scoring)
    svc = cross_validate(svc_model, X, y, cv=folds, scoring=scoring)
    dtr = cross_validate(dtr_model, X, y, cv=folds, scoring=scoring)
    rfc = cross_validate(rfc_model, X, y, cv=folds, scoring=scoring)
    gnb = cross_validate(gnb_model, X, y, cv=folds, scoring=scoring)
    xgb = cross_validate(xgb_model, X, y, cv=folds, scoring=scoring)
    gbm = cross_validate(gbm_model, X, y, cv=folds, scoring=scoring)
    hgbm = cross_validate(hgbm_model, X, y, cv=folds, scoring=scoring)
    lightgbm = cross_validate(lightgbm_model, X, y, cv=folds, scoring=scoring)

    # Create a data frame with the models perfoamnce metrics scores
    models_scores_table = pd.DataFrame({'Logistic Regression':[log['test_accuracy'].mean(),
                                                               log['test_precision'].mean(),
                                                               log['test_recall'].mean(),
                                                               log['test_f1_score'].mean()],
                                       
                                      'Support Vector Classifier':[svc['test_accuracy'].mean(),
                                                                   svc['test_precision'].mean(),
                                                                   svc['test_recall'].mean(),
                                                                   svc['test_f1_score'].mean()],
                                       
                                      'Decision Tree':[dtr['test_accuracy'].mean(),
                                                       dtr['test_precision'].mean(),
                                                       dtr['test_recall'].mean(),
                                                       dtr['test_f1_score'].mean()],
                                       
                                      'Random Forest':[rfc['test_accuracy'].mean(),
                                                       rfc['test_precision'].mean(),
                                                       rfc['test_recall'].mean(),
                                                       rfc['test_f1_score'].mean()],
                                       
                                      'Gaussian Naive Bayes':[gnb['test_accuracy'].mean(),
                                                              gnb['test_precision'].mean(),
                                                              gnb['test_recall'].mean(),
                                                              gnb['test_f1_score'].mean()],
                                       
                                       'XGBoost':[xgb['test_accuracy'].mean(),
                                                  xgb['test_precision'].mean(),
                                                  xgb['test_recall'].mean(),
                                                  xgb['test_f1_score'].mean()],
                                       
                                       'GradientBoostingClassifier':[gbm['test_accuracy'].mean(),
                                                                     gbm['test_precision'].mean(),
                                                                     gbm['test_recall'].mean(),
                                                                     gbm['test_f1_score'].mean()],
                                       
                                       'Hist GradientBoostingClassifier':[hgbm['test_accuracy'].mean(),
                                                                          hgbm['test_precision'].mean(),
                                                                          hgbm['test_recall'].mean(),
                                                                          hgbm['test_f1_score'].mean()],
                                       
                                       'LGBMClassifier':[lightgbm['test_accuracy'].mean(),
                                                         lightgbm['test_precision'].mean(),
                                                         lightgbm['test_recall'].mean(),
                                                         lightgbm['test_f1_score'].mean()]},
                                       
                                      index=['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    # Add 'Best Score' column
    models_scores_table['Best Score'] = models_scores_table.idxmax(axis=1)
    
    # Return models performance metrics scores data frame
    return(models_scores_table)
  
# Run models_evaluation function
models_evaluation(X_sample1, y_sample1, 5)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




