import sagemaker
import boto3
from sagemaker.dataset_definition.inputs import (
    AthenaDatasetDefinition,
    DatasetDefinition,
)
from sagemaker.workflow.clarify_check_step import (
    DataBiasCheckConfig,
    ClarifyCheckStep,
    ModelBiasCheckConfig,
    ModelExplainabilityCheckConfig,
    ClarifyCheckConfig,
)
from sagemaker.clarify import (
    DataConfig,
    BiasConfig,
    ModelConfig,
    ModelPredictedLabelConfig,
    SHAPConfig,
)
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.inputs import TrainingInput
from sagemaker.lambda_helper import Lambda
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    Processor,
    ScriptProcessor,
)
from sagemaker.workflow.check_job_config import CheckJobConfig
from sagemaker.workflow.clarify_check_step import ClarifyCheckStep, DataBiasCheckConfig
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.step_collections import EstimatorTransformer, RegisterModel
from sagemaker.workflow.steps import CacheConfig, ProcessingStep, Step, TrainingStep
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.utils import name_from_base
from sagemaker.inputs import CreateModelInput
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.model import Model

import uuid

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


def get_pipeline(
    role: str,
    pipeline_name: str,
    sagemaker_session: sagemaker.Session = None,
    **kwargs,
) -> Pipeline:
    
    base_job_prefix = kwargs["base_job_prefix"]
    bucket = kwargs["bucket"]
    prefix = kwargs["prefix"]
    label_name = kwargs["label_name"]
    features_names = kwargs["features_names"]
    model_package_group_name = kwargs["model_package_group_name"]
    customers_fg_name = kwargs["customers_fg_name"]
    script_create_dataset = kwargs["script_create_dataset"]
    script_evaluation = kwargs["script_evaluation"]
    
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    
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

    train_instance_count = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=1,
    )
    train_instance_type = ParameterString(
        name="TrainingInstance",
        default_value="ml.m5.xlarge",
    )


    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
        enum_values=[
            "PendingManualApproval",
            "Approved",
        ],
    )
    model_output = ParameterString(
        name="ModelOutputUrl",
        default_value=f"s3://{bucket}/{prefix}/model",
    )

    check_job_config = CheckJobConfig(
        role=role,
        instance_count=processing_instance_count,
        instance_type=processing_instance_type,
        volume_size_in_gb=120,
        sagemaker_session=sagemaker_session,
    )
    
    training_columns = [label_name] + features_names
    training_columns_string = ", ".join(f'"{c}"' for c in training_columns)
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
    SELECT DISTINCT {training_columns_string}
        FROM customer_table
        WHERE customer_table."rank" = 1
    """
    
    athena_data_path = "/opt/ml/processing/athena"

    create_dataset_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}-create-dataset",
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
                            "raw_dataset",
                        ],
                    ),
                    output_format="PARQUET",
                ),
            ),
        )
    ]
    
    create_dataset_step = ProcessingStep(
        name="CreateDataSet",
        processor=create_dataset_processor,
        inputs=data_sources,
        outputs=[
            ProcessingOutput(
                output_name="train_data",
                source="/opt/ml/processing/output/train",
                destination=Join(
                    on="/",
                    values=[
                        "s3:/",
                        bucket,
                        prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        "train_dataset",
                    ],
                ),
            ),
            ProcessingOutput(
                output_name="validation_data",
                source="/opt/ml/processing/output/validation",
                destination=Join(
                    on="/",
                    values=[
                        "s3:/",
                        bucket,
                        prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        "validation_dataset",
                    ],
                ),
            ),
            ProcessingOutput(
                output_name="test_data",
                source="/opt/ml/processing/output/test",
                destination=Join(
                    on="/",
                    values=[
                        "s3:/",
                        bucket,
                        prefix,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        "test_dataset",
                    ],
                ),
            ),
            ProcessingOutput(
                output_name="baseline_data",
                source="/opt/ml/processing/output/baseline",
                destination=Join(
                    on="/",
                    values=[
                        "s3:/",
                        bucket,
                        prefix,
                        "baseline",
                    ],
                ),
            ),
        ],
        job_arguments=[
            "--athena-data",
            athena_data_path,
        ],
        code=script_create_dataset,
        cache_config=cache_config,
    )
    
    # training step for generating model artifacts
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
    )
    xgb_train = sagemaker.estimator.Estimator(
        image_uri=image_uri,
        instance_type=train_instance_type,
        instance_count=train_instance_count,
        output_path=model_output,
        base_job_name=f"{base_job_prefix}-train",
        sagemaker_session=sagemaker_session,
        role=role,
        disable_profiler=True,
    )

    # Set some hyper parameters
    # https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html
    xgb_train.set_hyperparameters(
            max_depth=5,
            eta=0.2,
            gamma=4,
            min_child_weight=6,
            subsample=0.8,
            silent=0,
            objective="binary:logistic",
            num_round=100,
            eval_metric='auc'
        )

    step_train = TrainingStep(
        name="TrainModel",
        estimator=xgb_train,
        inputs={
            "train": TrainingInput(
                s3_data=create_dataset_step.properties.ProcessingOutputConfig.Outputs[
                    "train_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=create_dataset_step.properties.ProcessingOutputConfig.Outputs[
                    "validation_data"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
        cache_config=cache_config,

    )

    
    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=create_dataset_step.properties.ProcessingOutputConfig.Outputs[
                    "test_data"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation", source="/opt/ml/processing/evaluation"
            ),
        ],
        code=script_evaluation,
        property_files=[evaluation_report],
        cache_config=cache_config,
    )

    

    model = Model(
        image_uri=image_uri,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role,
    )

    inputs = CreateModelInput(
        instance_type="ml.m5.xlarge",
    )
    step_create_model = CreateModelStep(
        name="CustomerChurnCreateModel",
        model=model,
        inputs=inputs,
    )
    
    # clarify step
    check_job_config = CheckJobConfig(
                    role=role,
                    instance_type="ml.m5.xlarge",
                    instance_count=1,
                    sagemaker_session=sagemaker_session,
    )

    # explainability output path
    explainability_output_uri = "s3://{}/{}/clarify-explainability".format(bucket, prefix)

    data_config = DataConfig(
                    s3_data_input_path=create_dataset_step.properties.ProcessingOutputConfig.Outputs[
                        "train_data"
                    ].S3Output.S3Uri,
                    s3_output_path=explainability_output_uri,
                    label=label_name,
                    headers=[label_name] + features_names,
                    dataset_type="text/csv",
    )
    shap_config = SHAPConfig(
                    baseline=f"s3://{bucket}/{prefix}/baseline/baseline.csv",
                    num_samples=15,
                    agg_method="mean_abs",
                    save_local_shap_values=True,
    )
    model_config = ModelConfig(
                    model_name=step_create_model.properties.ModelName,
                    instance_type="ml.m5.xlarge",
                    instance_count=1,
                    accept_type="text/csv",
                    content_type="text/csv",
    )

    model_explainability_check_config = ModelExplainabilityCheckConfig(
        data_config=data_config,
        model_config=model_config,
        explainability_config=shap_config,
    )

    model_explainability_check_step = ClarifyCheckStep(
        name="ModelExplainabilityCheckStep",
        clarify_check_config=model_explainability_check_config,
        check_job_config=check_job_config,
        skip_check=True,
        register_new_baseline=True,
        cache_config=cache_config,
    )
    
    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"][
                    "S3Uri"
                ]
            ),
            content_type="application/json",
        )
    )
    step_register = RegisterModel(
        name="RegisterModel",
        estimator=xgb_train,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.t2.large", "ml.m5.large"],
        transform_instances=["ml.m5.large"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="binary_classification_metrics.roc_auc.value",
        ),
        right=0.93,
    )
    step_cond = ConditionStep(
        name="CheckEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register],
        else_steps=[],
    )
    
    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            processing_instance_type,
            train_instance_count,
            train_instance_type,
            model_approval_status,
            model_output
        ],
        steps=[create_dataset_step, step_train, step_eval, step_create_model, model_explainability_check_step, step_cond],
        sagemaker_session=sagemaker_session,
    )
    
    return pipeline