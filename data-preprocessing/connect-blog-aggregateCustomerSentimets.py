import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import *
from pyspark.sql import SparkSession, HiveContext
from pyspark.sql import Row
import datetime
from pyspark.sql.types import *
from pyspark.sql.functions import explode
import pyspark.sql.functions as F


args = getResolvedOptions(sys.argv, ["JOB_NAME"])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

#Define Database and Table names:
connect_database = "amazonconnect_db"
connectCTR_table = "cfn_connect_connect_mmnm_ctr"
contactLense_table = "cfn_connect_voice"
connect_Customer_table = "cfn_connect_customerdata"
output_path = "s3://amazon-connect-f5277b9142db/aggregated_customer_data_25052022/"



#Read contact lense data
datasource_contactLense  = glueContext.create_dynamic_frame.from_catalog(
    database = connect_database, 
    table_name = contactLense_table, 
    transformation_ctx = "datasource_contactLense"
)


#select only required field from Contactlens data and flatten the nested data type 
selectedfields = SelectFields.apply(datasource_contactLense, paths=['CustomerMetadata','Transcript'])
contactlense_df= selectedfields.toDF()
contactlense_df_exploded = contactlense_df.select(contactlense_df.CustomerMetadata, explode(contactlense_df.Transcript))

# read contact center data
datasource_ContactCenter  = glueContext.create_dynamic_frame.from_catalog(
    database = connect_database, 
    table_name = connectCTR_table, 
    transformation_ctx = "datasource_ContactCenter"
)

# select only required fields from CTR data
datasource_ContactCenter_requiredFields = SelectFields.apply(datasource_ContactCenter, paths=['ContactId','Attributes.customerid']).toDF()

#read Customer Data
datasource_customer  = glueContext.create_dynamic_frame.from_catalog(
    database = connect_database, 
    table_name = connect_Customer_table, 
    transformation_ctx = "datasource_customer")
customer_df=datasource_customer.toDF()


#Count Customer sentiments for each contact ID
sentiments_count = contactlense_df_exploded.filter(contactlense_df_exploded.col.ParticipantId == "CUSTOMER").groupBy("CustomerMetadata.ContactId", "col.Sentiment").count()
sentiments_count_bycategory = sentiments_count.groupBy("ContactId").pivot("Sentiment").agg(F.first("count"))


#Join conatctlense data with contact center data and group by Contactid count sum of each sentiments
aggregated_Sentiments_by_customer = sentiments_count_bycategory.join(datasource_ContactCenter_requiredFields, sentiments_count_bycategory.ContactId == datasource_ContactCenter_requiredFields.ContactId).select(datasource_ContactCenter_requiredFields["Attributes.customerid"],sentiments_count_bycategory["*"]).groupBy("customerid").agg(sum("MIXED"),sum("NEGATIVE"),sum("NEUTRAL"),sum("POSITIVE"))

#Join the current transformed dataset with Customer data to pull customer name and other customer attributes
aggregated_customerSentimentData = customer_df.join(aggregated_Sentiments_by_customer,aggregated_Sentiments_by_customer.customerid == customer_df.customerid).drop(aggregated_Sentiments_by_customer.customerid)

#Save result to S3
aggregated_customerSentimentData.write.mode("overwrite").json(output_path)


job.commit()
