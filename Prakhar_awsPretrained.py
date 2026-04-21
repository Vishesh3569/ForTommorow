import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
from datetime import datetime
import tarfile
import os

# Initialize role
try:
    role = sagemaker.get_execution_role()
except ValueError:
    role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/service-role/AmazonSageMaker-ExecutionRole"

sagemaker_session = sagemaker.Session()
s3 = boto3.client("s3")

bucket = "finalmlmpde"
pkl_key = "my_heart_model.pkl"   # from your structure

endpoint_name = f"heart-model-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# Step 1: Download existing .pkl from S3
print("Downloading .pkl file...")
local_pkl = "my_heart_model.pkl"
s3.download_file(bucket, pkl_key, local_pkl)

# Step 2: Create model.tar.gz
print("Creating model.tar.gz...")
with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add(local_pkl)

# Step 3: Upload correct artifact
print("Uploading model.tar.gz...")
model_key = "model.tar.gz"
s3.upload_file("model.tar.gz", bucket, model_key)

model_data_s3_uri = f"s3://{bucket}/{model_key}"
print("Model uploaded to:", model_data_s3_uri)

# Step 4: Deploy
print("Deploying to SageMaker...")
sklearn_model = SKLearnModel(
    model_data=model_data_s3_uri,
    role=role,
    entry_point="inference.py",
    framework_version="1.2-1",
    py_version="py3",
    sagemaker_session=sagemaker_session
)

predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name=endpoint_name
)

print("\n✅ Deployment successful!")
print("Endpoint:", endpoint_name)
