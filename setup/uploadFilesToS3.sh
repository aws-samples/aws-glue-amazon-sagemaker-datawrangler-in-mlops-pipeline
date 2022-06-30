#!/bin/bash
#set -x

s3path=` echo $1 | sed 's/\/$//'`

printf "will upload files to following S3 path:  $s3path\n"

ctr_data_path=$s3path/connect-ctr-data/2022/04/15/
conatctlens_data_path=$s3path/contactlens-data/Voice/2021/12/30/
customer_data_path=$s3path/customer-data/
etlscript_path=$s3path/glue-etl-script/

cf_template_path=$s3path/cf_templates/

#ls -lrt ./data/

#upload sample data files
aws s3 cp ./data/CTRdata/ConnectCTR-1-2022-04-15-06-03-11-cce4b6f2-5da2-4c91-92d5-9b7fd9e300f7 $ctr_data_path

if [ $? -ne 0 ];
then
 printf "*****************************************************\n"
 printf "ERROR: check if S3 bucket and prefix exists and user has permission\n"
 exit
fi 


aws s3 cp ./data/ContactLensdata/02f69e7b-e236-4c1c-a0cd-bf845361fa46_analysis_2021-12-30T08_07_03Z.json $conatctlens_data_path
aws s3 cp ./data/Customerdata/SFDCcustData.csv $customer_data_path

#upload Glue ETL script
aws s3 cp ./data-preprocessing/connect-blog-aggregateCustomerSentimets.py $etlscript_path

#upload CF Template
aws s3 cp ./data-preprocessing/connect-database-crawler-cftemplate.yaml $cf_template_path

printf "\n\n\n"
printf "#########################################################\n"
printf " Files uploaded to following paths\n"

printf "CTR Data                = $s3path/connect-ctr-data/\n"
printf "Contact Lens data       = $s3path/contactlens-data/Voice/\n"
printf "Customer Reference data = $customer_data_path\n"

printf "Glue ETL Script         = ${etlscript_path}connect-blog-aggregateCustomerSentimets.py\n"
printf "CF template             = ${cf_template_path}connect-database-crawler-cftemplate.yaml\n"


#End of Script

