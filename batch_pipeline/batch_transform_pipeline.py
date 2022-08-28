import sagemaker
import boto3
from sagemaker.dataset_definition.inputs import (
    AthenaDatasetDefinition,
    DatasetDefinition,
)

from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.lambda_helper import Lambda
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    Processor,
    ScriptProcessor,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.step_collections import EstimatorTransformer, RegisterModel
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, Step, TransformStep
from sagemaker.utils import name_from_base
from sagemaker.inputs import CreateModelInput, TransformInput
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.model import Model
from sagemaker.transformer import Transformer

from sagemaker.workflow.lambda_step import (
    LambdaStep,
    LambdaOutput,
    LambdaOutputTypeEnum,
)


def get_fg_info(fg_name: str, sagemaker_session: sagemaker.Session):
    
    boto_session = sagemaker_session.boto_session
    featurestore_runtime = sagemaker_session.sagemaker_featurestore_runtime_client
    feature_store_session = sagemaker.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_session.sagemaker_client,
        sagemaker_featurestore_runtime_client=featurestore_runtime,
    )
    fg = FeatureGroup(name=fg_name, sagemaker_session=feature_store_session)
    
    return fg.athena_query()


def get_batch_pipeline(
    role: str,
    pipeline_name: str,
    sagemaker_session: sagemaker.Session = None,
    **kwargs,
) -> Pipeline:
    
    base_job_prefix = kwargs["base_job_prefix"]
    bucket = kwargs["bucket"]
    prefix = kwargs["prefix"]
    features_names = kwargs["features_names"]
    model_package_group_name = kwargs["model_package_group_name"]
    customers_fg_name = kwargs["customers_fg_name"]
    script_create_batch_dataset = kwargs["script_create_batch_dataset"]
    lambda_role = kwargs["lambda_role"]
    lambda_function_name = kwargs["lambda_function_name"]
    lambda_script = kwargs["lambda_script"]
    lambda_handler = kwargs["lambda_handler"]
    
    
    session = boto3.Session()
    sagemaker_client = session.client(service_name="sagemaker")
    region=session.region_name
    
    # parameters for pipeline execution

    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1
    )
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )

    batch_output = ParameterString(
        name="BatchOutputUrl",
        default_value=f"s3://{bucket}/{prefix}/batch_output",
    )
    
    
    inference_columns = features_names
    inference_columns_string = ", ".join(f'"{c}"' for c in inference_columns)
    
    customers_fg_info = get_fg_info(
        customers_fg_name,
        sagemaker_session=sagemaker_session,
    )
    customers_fg = sagemaker_client.describe_feature_group(
        FeatureGroupName=customers_fg_name
    )
    customer_uid = customers_fg["RecordIdentifierFeatureName"]
    customer_et = customers_fg["EventTimeFeatureName"]


    query_string = f"""WITH customer_table AS (
        SELECT *,
            dense_rank() OVER (
                PARTITION BY "{customer_uid}"
                ORDER BY "{customer_et}" DESC,
                    "api_invocation_time" DESC,
                    "write_time" DESC
            ) AS "rank"
        FROM "{customers_fg_info.table_name}"
        WHERE NOT "is_deleted"
    )
    SELECT DISTINCT {inference_columns_string}
        FROM customer_table
        WHERE customer_table."rank" = 1
    """
    
    athena_data_path = "/opt/ml/processing/athena"

    create_batchdata_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.xlarge",
        instance_count=1,
        base_job_name=f"{base_job_prefix}-create-batch-dataset",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    data_sources = [
        ProcessingInput(
            input_name="athena_dataset",
            dataset_definition=DatasetDefinition(
                local_path=athena_data_path,
                data_distribution_type="FullyReplicated",
                athena_dataset_definition=AthenaDatasetDefinition(
                    catalog=customers_fg_info.catalog,
                    database=customers_fg_info.database,
                    query_string=query_string,
                    output_s3_uri=Join(
                        on="/",
                        values=[
                            "s3:/",
                            bucket,
                            prefix,
                            ExecutionVariables.PIPELINE_EXECUTION_ID,
                            "athena_data",
                        ],
                    ),
                    output_format="PARQUET",
                ),
            ),
        )
    ]
    
    step_batch_data = ProcessingStep(
        name="GetBatchData",
        processor=create_batchdata_processor,
        inputs=data_sources,
        outputs=[
            ProcessingOutput(
                output_name="batch_data",
                source="/opt/ml/processing/output/batch",
                destination=Join(
                    on="/",
                    values=[
                        "s3:/",
                        bucket,
                        prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        "batch_data",
                    ],
                ),
            ),
        ],
        job_arguments=[
            "--athena-data", athena_data_path,
        ],
        code=script_create_batch_dataset,
    )

    
    # Lambda helper class can be used to create the Lambda function
    func = Lambda(
        function_name=lambda_function_name,
        execution_role_arn=lambda_role,
        script=lambda_script,
        handler=lambda_handler,
        timeout=600,
        memory_size=128,
    )

    step_latest_model_fetch = LambdaStep(
        name="fetchLatestModel",
        lambda_func=func,
        inputs={
            "model_package_group_name": model_package_group_name,
        },
        outputs=[
            LambdaOutput(output_name="ModelUrl", output_type=LambdaOutputTypeEnum.String), 
            LambdaOutput(output_name="ImageUri", output_type=LambdaOutputTypeEnum.String), 
        ],
    )
    
    model = Model(
        image_uri=step_latest_model_fetch.properties.Outputs["ImageUri"],
        model_data=step_latest_model_fetch.properties.Outputs["ModelUrl"],
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    inputs = CreateModelInput(
        instance_type="ml.m5.xlarge",
    )
    step_create_model = CreateModelStep(
        name="CreateModel",
        model=model,
        inputs=inputs,
    )
    
    transformer = Transformer(
        model_name=step_create_model.properties.ModelName,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        output_path=batch_output,
        accept="text/csv",
        assemble_with="Line",
    )

    step_transform = TransformStep(
        name="Transform", 
        transformer=transformer, 
        inputs=TransformInput(
            data=step_batch_data.properties.ProcessingOutputConfig.Outputs[
                    "batch_data"
                ].S3Output.S3Uri,
            split_type="Line",
            content_type="text/csv",
            join_source="Input",
        ),
    )
    
    pipeline = Pipeline(
        name=pipeline_name,
        steps=[step_batch_data, step_latest_model_fetch, step_create_model, step_transform],
    )
    
    return pipeline