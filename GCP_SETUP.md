GCP setup steps for CI/CD with Artifact Registry and GKE

This document lists the minimum steps (commands and IAM) to prepare GCP for the GitHub Actions CI/CD pipeline that builds the FastAPI image and deploys to GKE.

1) Enable required APIs

   gcloud services enable artifactregistry.googleapis.com \
     containerregistry.googleapis.com \
     container.googleapis.com \
     compute.googleapis.com \
     iam.googleapis.com

2) Create Artifact Registry repository (docker format)

   gcloud artifacts repositories create REPO_NAME \
     --repository-format=docker \
     --location=LOCATION \
     --description="Docker repo for iris-api"

3) Create a GKE cluster

   gcloud container clusters create CLUSTER_NAME \
     --zone=ZONE_OR_REGION \
     --num-nodes=1

4) Create a service account for GitHub Actions

   gcloud iam service-accounts create gha-deployer --display-name="GHA Deployer"

   Grant required roles to the service account (project-level or resource-specific):

   - roles/artifactregistry.writer (push images)
   - roles/container.admin (manage GKE workloads)
   - roles/container.developer (optional, for interacting with clusters)
   - roles/storage.admin (if using GCS for other artifacts)

   Example:
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:gha-deployer@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/artifactregistry.writer"

   (Repeat for container.admin etc.)

5) Create service account key and add to GitHub secrets

   gcloud iam service-accounts keys create key.json \
     --iam-account=gha-deployer@PROJECT_ID.iam.gserviceaccount.com

   In GitHub repository settings -> Secrets, create these secrets:
   - GCP_SA_KEY: contents of key.json
   - GCP_PROJECT: your project id
   - ARTIFACT_REGISTRY_LOCATION: e.g. us-central1
   - ARTIFACT_REGISTRY_REPOSITORY: the repo name you created
   - GKE_CLUSTER: cluster name
   - GKE_LOCATION: cluster zone/region

6) Protect your production environment (optional, recommended)

   - In GitHub, create an Environment called "production" and configure Required reviewers so that deploy job requires manual approval.

Notes and tips:

- For Artifact Registry docker repositories the hostname is: ${LOCATION}-docker.pkg.dev
- Ensure the machine/build environment (or GitHub Actions) has permission to push to Artifact Registry; the service account must have Artifact Registry Writer role.
- If you want private networking, configure VPC, private cluster, and appropriate firewall rules.
- Consider using Workload Identity for GKE in production rather than service account keys.

Project-specific example values (from user):

- Artifact Registry base: us-central1-docker.pkg.dev/coherent-bliss-474210-q3/iris-repo
- GKE region: us-central1
- GKE endpoint / cluster IP (possible): 35.224.114.114
- MLflow server IP: 35.223.200.47 (the repo uses port 8100 in code; if MLflow listens on a different port update MLFLOW_TRACKING_URI accordingly)

Use the values above when creating these GitHub secrets:

- ARTIFACT_REGISTRY_LOCATION = us-central1
- ARTIFACT_REGISTRY_REPOSITORY = iris-repo
- GCP_PROJECT = coherent-bliss-474210-q3
- GKE_CLUSTER = (your cluster name; verify in GKE console)
- GKE_LOCATION = us-central1

If MLflow runs on a non-standard port, update the environment variable `MLFLOW_TRACKING_URI` in `k8s/iris-deployment.yaml` or set it via a Kubernetes `ConfigMap`/secret.
