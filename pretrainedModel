import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from sagemaker_studio import Project
from datetime import datetime
import tarfile
import os

# Initialize project
proj = Project()
role = proj.iam_role
region = boto3.Session().region_name
s3_client = boto3.client('s3')

# Configuration
source_bucket = "rohitpre"
source_key = "model.tar.gz/my_heart_model.pkl"
endpoint_name = f"heart-model-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Step 1: Download the pickle file
print("Downloading model file from S3...")
local_pkl_path = "/tmp/my_heart_model.pkl"
s3_client.download_file(source_bucket, source_key, local_pkl_path)

# Step 2: Create model.tar.gz with the pickle file
print("Packaging model as model.tar.gz...")
model_tar_path = "/tmp/model.tar.gz"
with tarfile.open(model_tar_path, "w:gz") as tar:
    tar.add(local_pkl_path, arcname="my_heart_model.pkl")

# Step 3: Upload model.tar.gz to S3
print("Uploading packaged model to S3...")
model_s3_key = f"models/heart-model/{datetime.now().strftime('%Y%m%d-%H%M%S')}/model.tar.gz"
s3_client.upload_file(model_tar_path, source_bucket, model_s3_key)
model_data_s3_uri = f"s3://{source_bucket}/{model_s3_key}"
print(f"Model uploaded to: {model_data_s3_uri}")

# Step 4: Create and deploy SKLearn Model
print("Creating SageMaker model and deploying endpoint...")
sklearn_model = SKLearnModel(
    model_data=model_data_s3_uri,
    role=role,
    framework_version="1.2-1",
    py_version="py3"
)

# Deploy the model
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print(f"\n✓ Endpoint created successfully!")
print(f"Endpoint Name: {endpoint_name}")
print(f"Endpoint ARN: {predictor.endpoint_arn}")
print(f"\nYou can now invoke this endpoint from your Lambda function using:")
print(f"  - Endpoint Name: {endpoint_name}")
print(f"  - Region: {region}")
