# AWS Architecture: AI-Powered Digital Twin for Indian Town

**Version:** 1.0  
**Date:** February 2026  
**Cloud Provider:** Amazon Web Services (AWS)  
**Region:** ap-south-1 (Mumbai)

---

## 1. High-Level AWS Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Layer                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Web App    │  │  Mobile App  │  │  Admin Panel │              │
│  │ (CloudFront) │  │  (Amplify)   │  │ (CloudFront) │              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
└─────────┼──────────────────┼──────────────────┼──────────────────────┘
          │                  │                  │
┌─────────┼──────────────────┼──────────────────┼──────────────────────┐
│         │         API Gateway & Authentication                       │
│         └──────────────────┴──────────────────┘                      │
│  ┌──────────────────────────────────────────────────────────┐       │
│  │  API Gateway (REST + WebSocket)                          │       │
│  │  + Cognito (Authentication)                              │       │
│  │  + WAF (Web Application Firewall)                        │       │
│  └──────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
          │
┌─────────┼─────────────────────────────────────────────────────────┐
│         │              Application Layer (EKS)                     │
│         ▼                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  Data API    │  │  ML Service  │  │  Simulation  │            │
│  │  (FastAPI)   │  │  (FastAPI)   │  │  (Python)    │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  Decision    │  │  Alert       │  │  Report      │            │
│  │  Support     │  │  Service     │  │  Generator   │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
          │
┌─────────┼─────────────────────────────────────────────────────────┐
│         │           Data Processing Layer                          │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  MSK (Managed Kafka) - Event Streaming                   │     │
│  └──────────────┬───────────────────────────────────────────┘     │
│                 │                                                  │
│  ┌──────────────▼──────────────┐  ┌──────────────────────┐       │
│  │  Kinesis Data Analytics     │  │  EMR (Spark)         │       │
│  │  (Stream Processing)        │  │  (Batch Processing)  │       │
│  └─────────────────────────────┘  └──────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
          │
┌─────────┼─────────────────────────────────────────────────────────┐
│         │              ML & AI Layer                               │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────┐     │
│  │  SageMaker                                               │     │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐        │     │
│  │  │  Training  │  │  Endpoints │  │  Feature   │        │     │
│  │  │  Jobs      │  │  (Models)  │  │  Store     │        │     │
│  │  └────────────┘  └────────────┘  └────────────┘        │     │
│  └──────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────┘
          │
┌─────────┼─────────────────────────────────────────────────────────┐
│         │                Storage Layer                             │
│         ▼                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │  RDS Aurora  │  │  DynamoDB    │  │  S3          │            │
│  │  (TimeSeries)│  │  (Documents) │  │  (Objects)   │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│  ┌──────────────┐  ┌──────────────┐                              │
│  │  ElastiCache │  │  Neptune     │                              │
│  │  (Redis)     │  │  (Graph)     │                              │
│  └──────────────┘  └──────────────┘                              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. AWS Services Mapping

### 2.1 Complete Service Mapping Table

| Component | AWS Service | Purpose | Justification |
|-----------|-------------|---------|---------------|
| **Frontend** |
| Web Dashboard | CloudFront + S3 | Static hosting, CDN | Low latency, global distribution |
| Mobile App | AWS Amplify | Mobile backend | Integrated auth, APIs, hosting |
| Admin Panel | CloudFront + S3 | Admin interface | Secure, fast delivery |
| **API & Auth** |
| API Gateway | API Gateway | REST + WebSocket APIs | Managed, scalable, integrated |
| Authentication | Cognito | User management | MFA, SSO, SAML support |
| Security | WAF + Shield | DDoS protection | Layer 7 protection |
| **Application** |
| Container Orchestration | EKS | Kubernetes cluster | Industry standard, portable |
| Microservices | EKS Pods | Application services | Scalable, isolated |
| Serverless Functions | Lambda | Event-driven tasks | Cost-effective, auto-scaling |
| **Data Ingestion** |
| IoT Data | IoT Core | MQTT broker | Managed, secure, scalable |
| Message Queue | MSK (Kafka) | Event streaming | High throughput, durable |
| API Ingestion | API Gateway | External APIs | Rate limiting, auth |
| **Data Processing** |
| Stream Processing | Kinesis Data Analytics | Real-time processing | SQL-based, managed |
| Batch Processing | EMR (Spark) | Large-scale ETL | Cost-effective, powerful |
| Workflow Orchestration | Step Functions | Complex workflows | Visual, serverless |
| **ML & AI** |
| Model Training | SageMaker Training | Train ML models | GPU instances, distributed |
| Model Hosting | SageMaker Endpoints | Model serving | Auto-scaling, A/B testing |
| Feature Store | SageMaker Feature Store | Feature management | Online/offline features |
| Model Registry | SageMaker Model Registry | Version control | Approval workflows |
| **Storage** |
| Time-Series DB | RDS Aurora PostgreSQL + TimescaleDB | Sensor data | High performance, scalable |
| Document DB | DynamoDB | Alerts, configs | NoSQL, serverless |
| Object Storage | S3 | Files, backups | Durable, cheap |
| Cache | ElastiCache (Redis) | Fast access | Sub-millisecond latency |
| Graph DB | Neptune | Causal relationships | Graph queries |
| **Analytics** |
| Data Warehouse | Redshift | Historical analytics | Columnar, fast queries |
| BI Tool | QuickSight | Dashboards, reports | Integrated, serverless |
| **Monitoring** |
| Metrics | CloudWatch | System metrics | Native integration |
| Logs | CloudWatch Logs | Centralized logging | Searchable, retention |
| Tracing | X-Ray | Distributed tracing | Request flow analysis |
| Alerting | SNS + CloudWatch Alarms | Notifications | Multi-channel |
| **Security** |
| Secrets | Secrets Manager | API keys, passwords | Rotation, encryption |
| Encryption | KMS | Key management | HSM-backed |
| Network | VPC | Isolation | Private subnets |
| IAM | IAM | Access control | Fine-grained permissions |
| **DevOps** |
| CI/CD | CodePipeline + CodeBuild | Deployment pipeline | Integrated, automated |
| IaC | CloudFormation / Terraform | Infrastructure as Code | Version controlled |
| Container Registry | ECR | Docker images | Private, secure |

---

## 3. Data Ingestion Architecture

### 3.1 Real-Time Data Ingestion Flow

```
IoT Sensors → AWS IoT Core → MSK (Kafka) → Kinesis Analytics → RDS/DynamoDB
     │              │              │              │
     │              │              │              └─→ ElastiCache (Cache)
     │              │              │
     │              │              └─→ S3 (Raw Data Archive)
     │              │
     │              └─→ Lambda (Data Validation)
     │
     └─→ IoT Rules Engine (Filtering)
```

**Detailed Flow**:

1. **IoT Sensors** (Traffic, Pollution, Water)
   - Protocol: MQTT over TLS 1.3
   - Frequency: 30 seconds to 5 minutes
   - Payload: JSON (< 1KB per message)

2. **AWS IoT Core**
   - Device authentication via X.509 certificates
   - Device shadows for offline handling
   - Rules engine for message routing
   ```sql
   -- IoT Rule Example
   SELECT * FROM 'sensors/traffic/+'
   WHERE congestion_level > 70
   ```

3. **MSK (Managed Kafka)**
   - Topics: `traffic`, `pollution`, `water`, `agriculture`, `market`
   - Partitions: 10 per topic (by location hash)
   - Replication factor: 3
   - Retention: 7 days

4. **Kinesis Data Analytics**
   - SQL-based stream processing
   - Windowing: Tumbling (1 min), Sliding (5 min)
   - Aggregations: AVG, MAX, COUNT
   ```sql
   -- Example: Traffic Aggregation
   CREATE OR REPLACE STREAM traffic_1min AS
   SELECT STREAM
       road_id,
       STEP(event_time BY INTERVAL '1' MINUTE) as window_time,
       AVG(vehicle_count) as avg_vehicles,
       MAX(congestion_level) as max_congestion
   FROM traffic_stream
   GROUP BY road_id, STEP(event_time BY INTERVAL '1' MINUTE);
   ```

5. **Storage**
   - Hot data (0-7 days): ElastiCache Redis
   - Warm data (7-90 days): RDS Aurora
   - Cold data (90+ days): S3 Glacier

### 3.2 Batch Data Ingestion Flow

```
External APIs → Lambda (Scheduler) → S3 (Landing) → Glue ETL → S3 (Processed) → Athena
     │                                      │
     │                                      └─→ EMR (Spark) → Feature Store
     │
     └─→ EventBridge (Scheduling)
```

**Sources**:
- IMD Weather API (every 15 minutes)
- AGMARKNET Market Data (hourly)
- Satellite Imagery (daily)
- Municipal Systems (daily batch files)

**ETL Process**:
```python
# AWS Glue ETL Job
import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext

# Read from S3
datasource = glueContext.create_dynamic_frame.from_catalog(
    database="digital_twin",
    table_name="raw_weather_data"
)

# Transform
transformed = datasource.apply_mapping([
    ("timestamp", "string", "timestamp", "timestamp"),
    ("temperature", "double", "temperature_celsius", "double"),
    ("rainfall", "double", "rainfall_mm", "double")
])

# Write to processed bucket
glueContext.write_dynamic_frame.from_options(
    frame=transformed,
    connection_type="s3",
    connection_options={"path": "s3://digital-twin-processed/weather/"},
    format="parquet"
)
```

---

## 4. ML Pipeline on AWS

### 4.1 Training Pipeline Architecture

```
S3 (Training Data) → SageMaker Processing → SageMaker Training → Model Registry
                            │                       │
                            │                       └─→ SageMaker Experiments
                            │
                            └─→ Feature Store
```

**Step-by-Step Flow**:

1. **Data Preparation** (SageMaker Processing)
```python
from sagemaker.processing import ScriptProcessor

processor = ScriptProcessor(
    role=role,
    image_uri='<ecr-image>',
    instance_type='ml.m5.xlarge',
    instance_count=2
)

processor.run(
    code='preprocessing.py',
    inputs=[
        ProcessingInput(
            source='s3://digital-twin-data/raw/',
            destination='/opt/ml/processing/input'
        )
    ],
    outputs=[
        ProcessingOutput(
            source='/opt/ml/processing/output',
            destination='s3://digital-twin-data/processed/'
        )
    ]
)
```

2. **Model Training** (SageMaker Training)
```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    role=role,
    instance_type='ml.p3.2xlarge',  # GPU instance
    instance_count=1,
    framework_version='1.13',
    py_version='py39',
    hyperparameters={
        'epochs': 100,
        'batch-size': 64,
        'learning-rate': 0.001
    }
)

estimator.fit({'training': 's3://digital-twin-data/processed/'})
```

3. **Model Registration**
```python
model_package = estimator.register(
    content_types=["application/json"],
    response_types=["application/json"],
    inference_instances=["ml.m5.xlarge"],
    transform_instances=["ml.m5.xlarge"],
    model_package_group_name="traffic-prediction-models",
    approval_status="PendingManualApproval"
)
```

### 4.2 Model Serving Architecture

```
API Gateway → Lambda (Routing) → SageMaker Endpoint → Model Container
                                         │
                                         └─→ Feature Store (Online)
```

**Endpoint Configuration**:
```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor

# Create model
model = Model(
    image_uri='<ecr-image>',
    model_data='s3://digital-twin-models/traffic-lstm-v2.3/model.tar.gz',
    role=role
)

# Deploy with auto-scaling
predictor = model.deploy(
    initial_instance_count=2,
    instance_type='ml.m5.xlarge',
    endpoint_name='traffic-prediction-prod'
)

# Configure auto-scaling
client = boto3.client('application-autoscaling')
client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=2,
    MaxCapacity=10
)

client.put_scaling_policy(
    PolicyName='traffic-prediction-scaling',
    ServiceNamespace='sagemaker',
    ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,  # Target 70% invocations per instance
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        }
    }
)
```


### 4.3 Feature Store Architecture

```
┌─────────────────────────────────────────────────────────┐
│         SageMaker Feature Store                          │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │  Offline Store (S3 + Glue Catalog)           │       │
│  │  - Historical features for training          │       │
│  │  - Parquet format                            │       │
│  │  - Queryable via Athena                      │       │
│  └──────────────────────────────────────────────┘       │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │  Online Store (DynamoDB)                     │       │
│  │  - Low-latency feature retrieval             │       │
│  │  - Sub-millisecond reads                     │       │
│  │  - Used for real-time inference              │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

**Feature Group Definition**:
```python
from sagemaker.feature_store.feature_group import FeatureGroup

traffic_feature_group = FeatureGroup(
    name="traffic-features",
    sagemaker_session=sagemaker_session
)

traffic_feature_group.load_feature_definitions(
    data_frame=traffic_df
)

traffic_feature_group.create(
    s3_uri=f"s3://digital-twin-features/traffic",
    record_identifier_name="road_id",
    event_time_feature_name="timestamp",
    role_arn=role,
    enable_online_store=True
)
```

**Feature Ingestion**:
```python
# Batch ingestion
traffic_feature_group.ingest(
    data_frame=traffic_df,
    max_workers=4,
    wait=True
)

# Real-time ingestion
record = [
    FeatureValue(feature_name='road_id', value_as_string='R001'),
    FeatureValue(feature_name='timestamp', value_as_string='2026-02-06T08:00:00Z'),
    FeatureValue(feature_name='avg_speed', value_as_string='35.5'),
    FeatureValue(feature_name='vehicle_count', value_as_string='450')
]

featurestore_runtime.put_record(
    FeatureGroupName='traffic-features',
    Record=record
)
```

**Feature Retrieval for Inference**:
```python
# Get latest features for prediction
response = featurestore_runtime.get_record(
    FeatureGroupName='traffic-features',
    RecordIdentifierValueAsString='R001'
)

features = {item['FeatureName']: item['ValueAsString'] 
            for item in response['Record']}
```

---

## 5. Simulation Engine on AWS

### 5.1 Architecture

```
API Gateway → Lambda (Orchestrator) → Step Functions → ECS Fargate (Simulation)
                                            │
                                            ├─→ S3 (Results)
                                            ├─→ DynamoDB (Status)
                                            └─→ SNS (Notifications)
```

**Step Functions Workflow**:
```json
{
  "Comment": "Digital Twin Simulation Workflow",
  "StartAt": "ValidateInput",
  "States": {
    "ValidateInput": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:ap-south-1:xxx:function:validate-simulation",
      "Next": "LoadHistoricalData"
    },
    "LoadHistoricalData": {
      "Type": "Task",
      "Resource": "arn:aws:states:::athena:startQueryExecution.sync",
      "Parameters": {
        "QueryString": "SELECT * FROM traffic_data WHERE date >= '2025-01-01'",
        "ResultConfiguration": {
          "OutputLocation": "s3://digital-twin-temp/query-results/"
        }
      },
      "Next": "RunSimulation"
    },
    "RunSimulation": {
      "Type": "Task",
      "Resource": "arn:aws:states:::ecs:runTask.sync",
      "Parameters": {
        "LaunchType": "FARGATE",
        "Cluster": "digital-twin-cluster",
        "TaskDefinition": "simulation-task",
        "NetworkConfiguration": {
          "AwsvpcConfiguration": {
            "Subnets": ["subnet-xxx"],
            "SecurityGroups": ["sg-xxx"],
            "AssignPublicIp": "DISABLED"
          }
        },
        "Overrides": {
          "ContainerOverrides": [{
            "Name": "simulation-container",
            "Environment": [
              {"Name": "SCENARIO", "Value.$": "$.scenario"},
              {"Name": "NUM_AGENTS", "Value.$": "$.num_agents"}
            ]
          }]
        }
      },
      "Next": "AnalyzeResults"
    },
    "AnalyzeResults": {
      "Type": "Task",
      "Resource": "arn:aws:lambda:ap-south-1:xxx:function:analyze-simulation",
      "Next": "NotifyCompletion"
    },
    "NotifyCompletion": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sns:publish",
      "Parameters": {
        "TopicArn": "arn:aws:sns:ap-south-1:xxx:simulation-complete",
        "Message.$": "$.results"
      },
      "End": true
    }
  }
}
```

**ECS Task Definition**:
```json
{
  "family": "simulation-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "4096",
  "memory": "16384",
  "containerDefinitions": [{
    "name": "simulation-container",
    "image": "<ecr-repo>/simulation:latest",
    "essential": true,
    "environment": [
      {"name": "S3_BUCKET", "value": "digital-twin-simulations"},
      {"name": "DYNAMODB_TABLE", "value": "simulation-status"}
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/simulation",
        "awslogs-region": "ap-south-1",
        "awslogs-stream-prefix": "ecs"
      }
    }
  }]
}
```

### 5.2 Monte Carlo Simulations (Parallel Execution)

**Using AWS Batch**:
```python
import boto3

batch = boto3.client('batch')

# Submit 1000 simulation jobs
job_ids = []
for i in range(1000):
    response = batch.submit_job(
        jobName=f'simulation-run-{i}',
        jobQueue='digital-twin-queue',
        jobDefinition='simulation-job-def',
        parameters={
            'scenario': 'odd_even_scheme',
            'run_id': str(i),
            'seed': str(i * 42)  # Different seed for each run
        }
    )
    job_ids.append(response['jobId'])

# Wait for all jobs to complete
waiter = batch.get_waiter('job_complete')
for job_id in job_ids:
    waiter.wait(jobs=[job_id])

# Aggregate results
results = []
for job_id in job_ids:
    result = s3.get_object(
        Bucket='digital-twin-simulations',
        Key=f'results/{job_id}.json'
    )
    results.append(json.loads(result['Body'].read()))

# Statistical analysis
mean_congestion_reduction = np.mean([r['congestion_reduction'] for r in results])
confidence_interval = np.percentile(
    [r['congestion_reduction'] for r in results],
    [2.5, 97.5]
)
```

---

## 6. Decision Support & Recommendation Engine

### 6.1 Architecture

```
User Request → API Gateway → Lambda (Orchestrator) → Step Functions
                                                           │
                                                           ├─→ SageMaker (Causal Inference)
                                                           ├─→ Lambda (Cost-Benefit Analysis)
                                                           ├─→ DynamoDB (Historical Interventions)
                                                           └─→ Lambda (Ranking Algorithm)
                                                                    │
                                                                    └─→ Response
```

**Recommendation Logic**:
```python
import boto3
import numpy as np
from typing import List, Dict

class RecommendationEngine:
    def __init__(self):
        self.dynamodb = boto3.resource('dynamodb')
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')
        self.interventions_table = self.dynamodb.Table('interventions')
        
    def get_recommendations(
        self,
        context: Dict,
        budget: float,
        constraints: Dict
    ) -> List[Dict]:
        # 1. Fetch candidate interventions
        interventions = self._get_candidate_interventions(context)
        
        # 2. Estimate causal impact for each
        for intervention in interventions:
            impact = self._estimate_causal_impact(
                intervention,
                context
            )
            intervention['estimated_impact'] = impact
        
        # 3. Calculate cost-effectiveness
        for intervention in interventions:
            intervention['cost_effectiveness'] = (
                intervention['estimated_impact']['benefit'] /
                intervention['cost']
            )
        
        # 4. Apply constraints
        feasible = self._apply_constraints(
            interventions,
            budget,
            constraints
        )
        
        # 5. Optimize portfolio
        optimal = self._optimize_portfolio(feasible, budget)
        
        # 6. Rank and return
        ranked = sorted(
            optimal,
            key=lambda x: x['cost_effectiveness'],
            reverse=True
        )
        
        return ranked[:10]  # Top 10
    
    def _estimate_causal_impact(
        self,
        intervention: Dict,
        context: Dict
    ) -> Dict:
        # Call SageMaker endpoint for causal inference
        payload = {
            'intervention': intervention['name'],
            'context': context
        }
        
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName='causal-inference-endpoint',
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        
        result = json.loads(response['Body'].read())
        return result
    
    def _optimize_portfolio(
        self,
        interventions: List[Dict],
        budget: float
    ) -> List[Dict]:
        # Multi-objective optimization using OR-Tools
        from ortools.linear_solver import pywraplp
        
        solver = pywraplp.Solver.CreateSolver('SCIP')
        
        # Decision variables
        x = {}
        for i, intervention in enumerate(interventions):
            x[i] = solver.BoolVar(f'x_{i}')
        
        # Objective: Maximize total impact
        objective = solver.Objective()
        for i, intervention in enumerate(interventions):
            objective.SetCoefficient(
                x[i],
                intervention['estimated_impact']['benefit']
            )
        objective.SetMaximization()
        
        # Budget constraint
        budget_constraint = solver.Constraint(0, budget)
        for i, intervention in enumerate(interventions):
            budget_constraint.SetCoefficient(
                x[i],
                intervention['cost']
            )
        
        # Solve
        status = solver.Solve()
        
        if status == pywraplp.Solver.OPTIMAL:
            selected = [
                interventions[i]
                for i in range(len(interventions))
                if x[i].solution_value() > 0.5
            ]
            return selected
        
        return []
```

**Lambda Function for Recommendation**:
```python
import json
import boto3

def lambda_handler(event, context):
    # Parse request
    body = json.loads(event['body'])
    current_context = body['context']
    budget = body.get('budget', 10000000)
    
    # Initialize engine
    engine = RecommendationEngine()
    
    # Get recommendations
    recommendations = engine.get_recommendations(
        context=current_context,
        budget=budget,
        constraints=body.get('constraints', {})
    )
    
    # Return response
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        },
        'body': json.dumps({
            'recommendations': recommendations,
            'total_cost': sum(r['cost'] for r in recommendations),
            'total_benefit': sum(r['estimated_impact']['benefit'] 
                                for r in recommendations)
        })
    }
```

---

## 7. Scalability Design

### 7.1 Auto-Scaling Configuration

**EKS Auto-Scaling**:
```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: data-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: data-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 25
        periodSeconds: 60
```

**Cluster Autoscaler**:
```yaml
# EKS Node Group Auto-Scaling
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig
metadata:
  name: digital-twin-cluster
  region: ap-south-1

managedNodeGroups:
  - name: application-nodes
    instanceType: m5.xlarge
    minSize: 3
    maxSize: 20
    desiredCapacity: 5
    volumeSize: 100
    labels:
      role: application
    tags:
      nodegroup-role: application
    iam:
      withAddonPolicies:
        autoScaler: true
        cloudWatch: true
        
  - name: ml-nodes
    instanceType: p3.2xlarge  # GPU instances
    minSize: 0
    maxSize: 5
    desiredCapacity: 1
    volumeSize: 200
    labels:
      role: ml-inference
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
```

**RDS Aurora Auto-Scaling**:
```python
import boto3

rds = boto3.client('rds')

# Enable Aurora Auto Scaling
rds.register_scalable_target(
    ServiceNamespace='rds',
    ResourceId='cluster:digital-twin-cluster',
    ScalableDimension='rds:cluster:ReadReplicaCount',
    MinCapacity=1,
    MaxCapacity=15
)

rds.put_scaling_policy(
    PolicyName='aurora-read-replica-scaling',
    ServiceNamespace='rds',
    ResourceId='cluster:digital-twin-cluster',
    ScalableDimension='rds:cluster:ReadReplicaCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'RDSReaderAverageCPUUtilization'
        },
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60
    }
)
```

### 7.2 Load Distribution

**Application Load Balancer Configuration**:
```json
{
  "LoadBalancerArn": "arn:aws:elasticloadbalancing:ap-south-1:xxx:loadbalancer/app/digital-twin-alb/xxx",
  "Scheme": "internet-facing",
  "IpAddressType": "ipv4",
  "SecurityGroups": ["sg-xxx"],
  "Subnets": ["subnet-xxx", "subnet-yyy", "subnet-zzz"],
  "TargetGroups": [
    {
      "TargetGroupArn": "arn:aws:elasticloadbalancing:ap-south-1:xxx:targetgroup/data-api/xxx",
      "HealthCheckEnabled": true,
      "HealthCheckPath": "/health",
      "HealthCheckIntervalSeconds": 30,
      "HealthyThresholdCount": 2,
      "UnhealthyThresholdCount": 3,
      "Matcher": {"HttpCode": "200"},
      "TargetType": "ip"
    }
  ],
  "Listeners": [
    {
      "Protocol": "HTTPS",
      "Port": 443,
      "Certificates": [{"CertificateArn": "arn:aws:acm:ap-south-1:xxx:certificate/xxx"}],
      "DefaultActions": [{
        "Type": "forward",
        "TargetGroupArn": "arn:aws:elasticloadbalancing:ap-south-1:xxx:targetgroup/data-api/xxx"
      }]
    }
  ]
}
```

**CloudFront Distribution**:
```json
{
  "DistributionConfig": {
    "CallerReference": "digital-twin-2026",
    "Origins": [{
      "Id": "S3-digital-twin-frontend",
      "DomainName": "digital-twin-frontend.s3.ap-south-1.amazonaws.com",
      "S3OriginConfig": {
        "OriginAccessIdentity": "origin-access-identity/cloudfront/xxx"
      }
    }],
    "DefaultCacheBehavior": {
      "TargetOriginId": "S3-digital-twin-frontend",
      "ViewerProtocolPolicy": "redirect-to-https",
      "AllowedMethods": ["GET", "HEAD", "OPTIONS"],
      "CachedMethods": ["GET", "HEAD"],
      "Compress": true,
      "DefaultTTL": 86400,
      "MinTTL": 0,
      "MaxTTL": 31536000
    },
    "PriceClass": "PriceClass_All",
    "Enabled": true,
    "ViewerCertificate": {
      "ACMCertificateArn": "arn:aws:acm:us-east-1:xxx:certificate/xxx",
      "SSLSupportMethod": "sni-only",
      "MinimumProtocolVersion": "TLSv1.2_2021"
    }
  }
}
```

