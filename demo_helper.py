import time

import boto3


def delete_project_resources(
    sagemaker_boto_client,
    endpoint_name=None,
    pipeline_name=None,
    mpg_name=None,
    prefix='sagemaker/DEMO-xgboost-customer-churn-connect',
    fg_name=None,
    delete_s3_objects=False,
    bucket_name=None,
):
    """Delete AWS resources created during demo.
    Keyword arguments:
    sagemaker_boto_client -- boto3 client for SageMaker used for demo (REQUIRED)
    endpoint_name     -- resource name of the model inference endpoint (default None)
    pipeline_name     -- resource name of the SageMaker Pipeline (default None)
    mpg_name          -- model package group name (default None)
    prefix            -- s3 prefix or directory for the demo (default 'sagemaker/DEMO-xgboost-customer-churn-connect')
    delete_s3_objects -- delete all s3 objects in the demo directory (default False)
    bucket_name       -- name of bucket created for demo (default None)
    """

    if endpoint_name is not None:
        try:
            sagemaker_boto_client.delete_endpoint(EndpointName=endpoint_name)
            print(f"Deleted endpoint: {endpoint_name}")
        except Exception as e:
            if "Could not find endpoint" in e.response.get("Error", {}).get("Message"):
                pass
            else:
                raise (e)

    if pipeline_name is not None:
        try:
            sagemaker_boto_client.delete_pipeline(PipelineName=pipeline_name)
            print(f"\nDeleted pipeline: {pipeline_name}")
        except Exception as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFound":
                pass
            else:
                raise (e)

    if mpg_name is not None:
        model_packages = sagemaker_boto_client.list_model_packages(ModelPackageGroupName=mpg_name)[
            "ModelPackageSummaryList"
        ]
        for mp in model_packages:
            sagemaker_boto_client.delete_model_package(ModelPackageName=mp["ModelPackageArn"])
            print(f"\nDeleted model package: {mp['ModelPackageArn']}")
            time.sleep(1)

        try:
            sagemaker_boto_client.delete_model_package_group(ModelPackageGroupName=mpg_name)
            print(f"\nDeleted model package group: {mpg_name}")
        except Exception as e:
            if "does not exist" in e.response.get("Error", {}).get("Message"):
                pass
            else:
                raise (e)

    models = sagemaker_boto_client.list_models(NameContains=fg_name, MaxResults=50)["Models"]
    print("\n")
    for m in models:
        sagemaker_boto_client.delete_model(ModelName=m["ModelName"])
        print(f"Deleted model: {m['ModelName']}")
        time.sleep(1)

    feature_groups = sagemaker_boto_client.list_feature_groups(NameContains=fg_name)[
        "FeatureGroupSummaries"
    ]
    print(feature_groups)
    for fg in feature_groups:
        sagemaker_boto_client.delete_feature_group(FeatureGroupName=fg["FeatureGroupName"])
        print(f"Deleted feature group: {fg['FeatureGroupName']}")
        time.sleep(1)

    if delete_s3_objects == True and bucket_name is not None:
        s3 = boto3.resource("s3")
        bucket = s3.Bucket(bucket_name)
        bucket.objects.filter(Prefix=f"{prefix}/").delete()
        print(f"\nDeleted contents of {bucket_name}/{prefix}")