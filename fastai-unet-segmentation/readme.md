# FastAI Semantic Segmentation on an AWS Ground Truth Dataset

This example shows a semantic segmentation model trained using the results of a labelling job from [AWS Ground Truth](https://aws.amazon.com/sagemaker/groundtruth/).

To use:

- Install requirements: `pip install -r requirements.txt`
- Make sure you're running your training script from a machine with AWS credentials that can read from your S3 bucket.
- Update the `AwsGroundTruthFetcher` initialization with the path to your AWS Ground Truth manifest file.
- Update the `mask_mappings` list with the classes in your semantic segmentation task.
- Run!
