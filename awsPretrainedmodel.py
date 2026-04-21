import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from datetime import datetime
import tarfile
import os

# Initialize standard SageMaker session and role
try:
    role = sagemaker.get_execution_role()
except ValueError:
    # If running outside of SageMaker (e.g., local machine), hardcode your IAM role ARN
    role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
s3_client = boto3.client('s3')

# Configuration
source_bucket = "vishesh35"
# Verify this key. S3 paths rarely contain "model.tar.gz/" as a directory.
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
    entry_point="inference.py",  # Required by SageMaker Scikit-Learn
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sagemaker_session
)

# Deploy the model
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print(f"\n✓ Endpoint created successfully!")
print(f"Endpoint Name: {endpoint_name}")
