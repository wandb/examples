import os
import os.path
import boto3
import json
from urllib.parse import urlparse

class AwsGroundTruthFetcher():
    """Read an AWS Ground Truth manifest file, and download all the source and
    result files referenced within it to a local staging directory.
    """
    def __init__(self, manifest_s3_url, job_name, working_dir='/tmp/working'):
        # Job name is used in the AWS manifest to prefix key data
        self.job_name = job_name
        # Should be in form s3://bucket/path
        self.manifest_url = urlparse(manifest_s3_url)
        self.working_dir = working_dir
        self.s3 = boto3.client('s3')

    def sync_down(self, s3_url, local_subpath=None):
        "Download s3 file to local file system."
        if not local_subpath:
            # Drop leading slash
            local_subpath = s3_url.path[1:]
        local_path = os.path.join(self.working_dir, local_subpath)
        local_dir = os.path.dirname(local_path)
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        if not os.path.exists(local_path):
            print('-  downloading {}'.format(s3_url.path[1:]))
            self.s3.download_file(s3_url.netloc, s3_url.path[1:], local_path)
        return local_path

    def get_manifest_lines(self):
        "Get manifest and return filtered lines."
        manifest_path = os.path.join(self.working_dir, 'manifest.jsonl')
        manifest_path = self.sync_down(self.manifest_url)
        manifest_lines = open(manifest_path).read().split('\n')
        manifest_lines = filter(bool, manifest_lines)
        manifest_lines = map(lambda l: json.loads(l), manifest_lines)
        manifest_lines = filter(lambda l: self.filter_manifest_line(l),
            manifest_lines)
        return manifest_lines

    def filter_manifest_line(self, line):
        "Return whether a manifest line should be included."
        source_ref = line['source-ref']
        job_metadata_key = '{}-ref-metadata'.format(self.job_name)
        job_metadata = line[job_metadata_key]
        if 'failure-reason' in job_metadata:
            print('!  skipping failed item {}: {}'.format(source_ref,
                job_metadata['failure-reason']))
            return False
        if source_ref.endswith('.webp'):
            print('!  skipping webp item {}'.format(source_ref))
            return False
        return True

    def fetch(self):
        "Fetch all items from the manifest and return local paths."
        manifest_lines = self.get_manifest_lines()
        for line in manifest_lines:
            source_url = urlparse(line['source-ref'])
            source_path = self.sync_down(source_url)
            result_key = '{}-ref'.format(self.job_name)
            result_ref = line[result_key]
            result_url = urlparse(result_ref)
            result_path = self.sync_down(result_url)
            yield { 'source': source_path, 'result': result_path }
