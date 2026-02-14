# Design Document: AI-Powered Digital Twin for Indian Town

**Version:** 1.0  
**Date:** February 2026  
**Status:** Draft  
**Owner:** Engineering & Architecture Team

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Design](#2-architecture-design)
3. [Data Architecture](#3-data-architecture)
4. [ML Model Architecture](#4-ml-model-architecture)
5. [API Design](#5-api-design)
6. [Security Architecture](#6-security-architecture)
7. [Deployment Architecture](#7-deployment-architecture)
8. [Technology Stack](#8-technology-stack)

---

## 1. System Overview

### 1.1 High-Level Architecture

The Digital Twin system follows a layered microservices architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     Presentation Layer                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Web App    │  │  Mobile App  │  │  Admin Panel │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Authentication │ Rate Limiting │ Load Balancing     │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Application Services                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │  Data    │ │   ML     │ │Simulation│ │ Decision │      │
│  │  API     │ │  Models  │ │  Engine  │ │ Support  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Data Processing Layer                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ Stream   │ │  Batch   │ │ Feature  │ │  Model   │      │
│  │Processing│ │Processing│ │  Store   │ │ Training │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                     Data Ingestion Layer                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │  Kafka   │ │   IoT    │ │   API    │ │  File    │      │
│  │ Streams  │ │ Gateway  │ │Connectors│ │ Ingestion│      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────────┐
│                       Storage Layer                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │TimescaleDB│ │ MongoDB  │ │   S3     │ │  Redis   │      │
│  │(Time-series)│(Documents)│ (Objects) │ (Cache)    │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Design Principles

**1. Microservices Architecture**
- Each domain (traffic, pollution, water, agriculture) is a separate service
- Services communicate via REST APIs and message queues
- Independent scaling and deployment
- Fault isolation

**2. Event-Driven Architecture**
- Real-time data flows through Kafka event streams
- Services react to events asynchronously
- Decoupled components
- Scalable data processing

**3. API-First Design**
- All functionality exposed via APIs
- OpenAPI specification for all endpoints
- Versioned APIs for backward compatibility
- Consistent error handling

**4. Cloud-Native**
- Containerized services (Docker)
- Orchestrated with Kubernetes
- Auto-scaling based on load
- Infrastructure as Code (Terraform)

**5. Security by Design**
- Zero-trust architecture
- Encryption everywhere
- Least privilege access
- Defense in depth

**6. Data-Centric**
- Data quality as first-class concern
- Immutable data storage
- Audit trails for all changes
- Data lineage tracking

---

## 2. Architecture Design

### 2.1 Component Architecture

#### 2.1.1 Data Ingestion Service

**Purpose**: Collect data from multiple sources and publish to Kafka

**Components**:

```
┌─────────────────────────────────────────────────────────┐
│           Data Ingestion Service                         │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   IoT MQTT   │  │  REST API    │  │  File Poller │ │
│  │   Listener   │  │  Endpoints   │  │              │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
│         │                  │                  │         │
│         └──────────────────┼──────────────────┘         │
│                            │                            │
│                   ┌────────▼────────┐                   │
│                   │  Data Validator │                   │
│                   │  & Transformer  │                   │
│                   └────────┬────────┘                   │
│                            │                            │
│                   ┌────────▼────────┐                   │
│                   │ Kafka Producer  │                   │
│                   └────────┬────────┘                   │
└────────────────────────────┼────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Kafka Topics   │
                    │  - traffic      │
                    │  - pollution    │
                    │  - water        │
                    │  - agriculture  │
                    │  - market       │
                    └─────────────────┘
```

**Key Design Decisions**:

1. **Protocol Support**:
   - MQTT for IoT sensors (lightweight, pub-sub)
   - REST API for external systems
   - File polling for batch data (CSV, Excel)
   - WebSocket for real-time streams

2. **Data Validation**:
   - Schema validation using JSON Schema
   - Range checks (e.g., temperature -10°C to 50°C)
   - Duplicate detection (time window: 10 seconds)
   - Anomaly flagging (>3 std dev from mean)

3. **Error Handling**:
   - Retry with exponential backoff (3 attempts)
   - Dead letter queue for failed messages
   - Alert on repeated failures
   - Graceful degradation (continue with partial data)

4. **Scalability**:
   - Horizontal scaling (multiple instances)
   - Partitioned Kafka topics by sensor location
   - Connection pooling for database writes
   - Rate limiting per data source

**Technology Choices**:
- Language: Python (asyncio for concurrent connections)
- MQTT Broker: Eclipse Mosquitto or AWS IoT Core
- Message Queue: Apache Kafka
- Monitoring: Prometheus metrics for throughput, latency, errors

#### 2.1.2 Stream Processing Service

**Purpose**: Real-time data processing and aggregation

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│         Stream Processing (Apache Flink)                 │
│                                                          │
│  Kafka Topics                                            │
│       │                                                  │
│       ▼                                                  │
│  ┌──────────────────────────────────────────────┐       │
│  │  Flink Source (Kafka Consumer)               │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Windowing & Aggregation                     │       │
│  │  - 1-minute tumbling windows                 │       │
│  │  - 5-minute sliding windows                  │       │
│  │  - Session windows for events                │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Enrichment                                  │       │
│  │  - Join with reference data                 │       │
│  │  - Geospatial lookups                       │       │
│  │  - Weather correlation                      │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Complex Event Processing                    │       │
│  │  - Pattern detection                         │       │
│  │  - Anomaly detection                         │       │
│  │  - Alert generation                          │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Flink Sinks                                 │       │
│  │  - TimescaleDB (time-series)                 │       │
│  │  - Redis (cache)                             │       │
│  │  - Kafka (processed events)                  │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

**Processing Pipelines**:

1. **Traffic Processing**:
   ```python
   traffic_stream
     .keyBy(lambda x: x['sensor_id'])
     .window(TumblingEventTimeWindows.of(Time.minutes(1)))
     .aggregate(TrafficAggregator())  # avg speed, vehicle count
     .filter(lambda x: x['congestion_level'] > 70)  # Alert threshold
     .addSink(AlertSink())
   ```

2. **Pollution Processing**:
   ```python
   pollution_stream
     .keyBy(lambda x: x['location'])
     .window(SlidingEventTimeWindows.of(Time.minutes(5), Time.minutes(1)))
     .apply(AQICalculator())  # Compute AQI from pollutants
     .join(weather_stream)  # Correlate with weather
     .addSink(TimescaleDBSink())
   ```

3. **Water Processing**:
   ```python
   water_stream
     .keyBy(lambda x: x['zone'])
     .window(TumblingEventTimeWindows.of(Time.hours(1)))
     .aggregate(WaterConsumptionAggregator())
     .process(LeakDetectionFunction())  # Detect anomalies
     .addSink(AlertSink())
   ```

**Key Design Decisions**:

1. **Windowing Strategy**:
   - Tumbling windows for regular aggregations (non-overlapping)
   - Sliding windows for moving averages
   - Session windows for event sequences (e.g., traffic incidents)

2. **State Management**:
   - RocksDB for state backend (disk-based, scalable)
   - Checkpointing every 5 minutes
   - Savepoints for version upgrades

3. **Exactly-Once Semantics**:
   - Kafka transactions for source
   - Two-phase commit for sinks
   - Idempotent operations where possible

4. **Backpressure Handling**:
   - Flink's built-in backpressure mechanism
   - Buffer sizing based on throughput
   - Alert on sustained backpressure

**Technology Choices**:
- Framework: Apache Flink (mature, exactly-once semantics)
- Alternative: Apache Spark Structured Streaming
- State: RocksDB
- Deployment: Flink on Kubernetes

#### 2.1.3 ML Model Service

**Purpose**: Serve ML model predictions via API

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│              ML Model Service                            │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │  FastAPI Application                         │       │
│  │  ┌────────────────────────────────────────┐  │       │
│  │  │  POST /predict/traffic                 │  │       │
│  │  │  POST /predict/pollution               │  │       │
│  │  │  POST /predict/water                   │  │       │
│  │  │  POST /predict/crop-yield              │  │       │
│  │  │  POST /predict/market-price            │  │       │
│  │  └────────────────────────────────────────┘  │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Model Registry (MLflow)                     │       │
│  │  - Model versioning                          │       │
│  │  - A/B testing                               │       │
│  │  - Canary deployments                        │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Feature Store (Feast)                       │       │
│  │  - Online features (Redis)                   │       │
│  │  - Offline features (S3)                     │       │
│  │  - Feature versioning                        │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Model Inference                             │       │
│  │  ┌────────────┐  ┌────────────┐             │       │
│  │  │  Traffic   │  │ Pollution  │             │       │
│  │  │  LSTM      │  │ XGBoost    │             │       │
│  │  └────────────┘  └────────────┘             │       │
│  │  ┌────────────┐  ┌────────────┐             │       │
│  │  │   Water    │  │   Crop     │             │       │
│  │  │  Prophet   │  │   RF       │             │       │
│  │  └────────────┘  └────────────┘             │       │
│  └──────────────────────────────────────────────┘       │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │  Model Monitoring                            │       │
│  │  - Prediction latency                        │       │
│  │  - Model drift detection                     │       │
│  │  - Feature drift detection                   │       │
│  │  - Prediction distribution                   │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

**API Design Example**:

```python
# POST /predict/traffic
{
  "request_id": "uuid",
  "location": {
    "road_id": "R001",
    "coordinates": [23.0225, 72.5714]
  },
  "forecast_horizon": "2h",  # 2h, 6h, 12h, 24h
  "include_confidence": true
}

# Response
{
  "request_id": "uuid",
  "predictions": [
    {
      "timestamp": "2026-02-06T10:00:00Z",
      "traffic_volume": 450,
      "average_speed": 35.5,
      "congestion_level": 65,
      "confidence_interval": {
        "lower": 400,
        "upper": 500,
        "confidence": 0.95
      }
    },
    ...
  ],
  "model_version": "traffic-lstm-v2.3",
  "inference_time_ms": 245
}
```

**Key Design Decisions**:

1. **Model Serving Strategy**:
   - Synchronous API for real-time predictions
   - Batch API for bulk predictions
   - Caching of frequent predictions (5-minute TTL)
   - Model warm-up on startup

2. **Feature Engineering**:
   - Online features from Redis (low latency)
   - Feature transformation in service
   - Feature validation before inference
   - Feature importance logging

3. **Model Versioning**:
   - Semantic versioning (major.minor.patch)
   - Shadow mode for new models (parallel prediction)
   - A/B testing with traffic splitting
   - Automated rollback on accuracy drop

4. **Performance Optimization**:
   - Model quantization (FP32 → FP16)
   - Batch inference (max batch size: 32)
   - GPU acceleration (NVIDIA T4 or A10)
   - Model compilation (TorchScript, ONNX)

**Technology Choices**:
- API Framework: FastAPI (async, high performance)
- Model Registry: MLflow
- Feature Store: Feast
- Model Serving: TorchServe or custom
- Monitoring: Prometheus + Grafana


#### 2.1.4 Simulation Engine Service

**Purpose**: Run agent-based simulations for scenario testing

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│           Simulation Engine Service                      │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │  Simulation API                              │       │
│  │  POST /simulate/scenario                     │       │
│  │  GET  /simulate/results/{id}                 │       │
│  │  POST /simulate/whatif                       │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Scenario Manager                            │       │
│  │  - Baseline scenario                         │       │
│  │  - High growth scenario                      │       │
│  │  - Climate stress scenario                   │       │
│  │  - Policy intervention scenarios             │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Agent-Based Model (MESA)                    │       │
│  │  ┌────────────┐  ┌────────────┐             │       │
│  │  │  Citizens  │  │  Vehicles  │             │       │
│  │  │  (10K)     │  │  (5K)      │             │       │
│  │  └────────────┘  └────────────┘             │       │
│  │  ┌────────────┐  ┌────────────┐             │       │
│  │  │ Businesses │  │   Farms    │             │       │
│  │  │  (500)     │  │  (200)     │             │       │
│  │  └────────────┘  └────────────┘             │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Environment Model                           │       │
│  │  - Road network graph                        │       │
│  │  - Water distribution network                │       │
│  │  - Pollution dispersion model                │       │
│  │  - Market equilibrium model                  │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Monte Carlo Engine                          │       │
│  │  - Parallel execution (1000 runs)            │       │
│  │  - Uncertainty quantification                │       │
│  │  - Sensitivity analysis                      │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Results Aggregation & Visualization         │       │
│  │  - Statistical summaries                     │       │
│  │  - Confidence intervals                      │       │
│  │  - Scenario comparisons                      │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

**Simulation Workflow**:

1. **Initialization**:
   ```python
   # Load historical data for calibration
   historical_data = load_data(start_date, end_date)
   
   # Initialize agents
   citizens = [Citizen(id, location, attributes) for id in range(10000)]
   vehicles = [Vehicle(id, type, location) for id in range(5000)]
   
   # Initialize environment
   environment = Environment(
       road_network=load_road_network(),
       water_network=load_water_network(),
       weather=WeatherGenerator(historical_patterns)
   )
   
   # Create simulation
   simulation = DigitalTwinSimulation(
       agents=citizens + vehicles + businesses + farms,
       environment=environment,
       time_step=3600  # 1 hour in seconds
   )
   ```

2. **Execution**:
   ```python
   # Run simulation
   for step in range(num_steps):
       # Update agents
       for agent in simulation.agents:
           agent.update(simulation.environment)
       
       # Update environment
       simulation.environment.update(simulation.agents)
       
       # Collect metrics
       simulation.collect_metrics()
       
       # Check termination conditions
       if simulation.should_terminate():
           break
   ```

3. **Analysis**:
   ```python
   # Aggregate results
   results = {
       'avg_traffic_congestion': np.mean(traffic_metrics),
       'avg_pollution': np.mean(pollution_metrics),
       'water_shortage_days': count_shortage_days(),
       'citizen_satisfaction': np.mean(satisfaction_scores),
       'economic_productivity': calculate_productivity()
   }
   
   # Compare with baseline
   improvement = (results - baseline) / baseline * 100
   ```

**Key Design Decisions**:

1. **Agent Modeling**:
   - Heterogeneous agents with different behaviors
   - Decision-making based on utility maximization
   - Learning from past experiences
   - Social interactions and influence

2. **Scalability**:
   - Spatial partitioning for efficient neighbor queries
   - Parallel agent updates (thread-safe)
   - Incremental environment updates
   - Checkpointing for long simulations

3. **Calibration**:
   - Parameter estimation from historical data
   - Validation against known outcomes
   - Sensitivity analysis for key parameters
   - Uncertainty quantification

4. **Performance**:
   - Vectorized operations (NumPy)
   - JIT compilation (Numba)
   - Multi-core processing
   - GPU acceleration for large-scale simulations

**Technology Choices**:
- Framework: MESA (Python agent-based modeling)
- Parallelization: Ray for distributed computing
- Visualization: Matplotlib, Plotly
- Deployment: Kubernetes with GPU nodes

---

## 3. Data Architecture

### 3.1 Data Model

#### 3.1.1 Time-Series Data (TimescaleDB)

**Traffic Measurements**:
```sql
CREATE TABLE traffic_measurements (
    time TIMESTAMPTZ NOT NULL,
    sensor_id VARCHAR(50) NOT NULL,
    road_id VARCHAR(50) NOT NULL,
    location GEOGRAPHY(POINT, 4326),
    vehicle_count INTEGER,
    average_speed FLOAT,
    occupancy_percentage FLOAT,
    congestion_level INTEGER,  -- 0-100
    metadata JSONB,
    PRIMARY KEY (time, sensor_id)
);

-- Hypertable for automatic partitioning
SELECT create_hypertable('traffic_measurements', 'time');

-- Indexes
CREATE INDEX idx_traffic_road ON traffic_measurements (road_id, time DESC);
CREATE INDEX idx_traffic_location ON traffic_measurements USING GIST (location);

-- Continuous aggregates for performance
CREATE MATERIALIZED VIEW traffic_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    road_id,
    AVG(vehicle_count) AS avg_vehicle_count,
    AVG(average_speed) AS avg_speed,
    MAX(congestion_level) AS max_congestion
FROM traffic_measurements
GROUP BY hour, road_id;
```

**Pollution Measurements**:
```sql
CREATE TABLE pollution_measurements (
    time TIMESTAMPTZ NOT NULL,
    station_id VARCHAR(50) NOT NULL,
    location GEOGRAPHY(POINT, 4326),
    pm25 FLOAT,  -- μg/m³
    pm10 FLOAT,
    co2 FLOAT,   -- ppm
    nox FLOAT,   -- ppb
    so2 FLOAT,
    o3 FLOAT,
    aqi INTEGER,  -- 0-500
    aqi_category VARCHAR(20),  -- Good, Moderate, Poor, etc.
    temperature FLOAT,
    humidity FLOAT,
    metadata JSONB,
    PRIMARY KEY (time, station_id)
);

SELECT create_hypertable('pollution_measurements', 'time');

-- Retention policy: Keep raw data for 90 days
SELECT add_retention_policy('pollution_measurements', INTERVAL '90 days');

-- Compression after 7 days
ALTER TABLE pollution_measurements SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'station_id'
);

SELECT add_compression_policy('pollution_measurements', INTERVAL '7 days');
```

**Water Measurements**:
```sql
CREATE TABLE water_measurements (
    time TIMESTAMPTZ NOT NULL,
    meter_id VARCHAR(50) NOT NULL,
    zone_id VARCHAR(50) NOT NULL,
    location GEOGRAPHY(POINT, 4326),
    consumption_liters FLOAT,
    flow_rate FLOAT,  -- liters/second
    pressure_bar FLOAT,
    meter_type VARCHAR(20),  -- residential, commercial, industrial
    metadata JSONB,
    PRIMARY KEY (time, meter_id)
);

SELECT create_hypertable('water_measurements', 'time');

-- Zone-level aggregation
CREATE MATERIALIZED VIEW water_zone_hourly
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    zone_id,
    SUM(consumption_liters) AS total_consumption,
    AVG(pressure_bar) AS avg_pressure,
    COUNT(DISTINCT meter_id) AS active_meters
FROM water_measurements
GROUP BY hour, zone_id;
```

#### 3.1.2 Document Data (MongoDB)

**Alerts Collection**:
```javascript
{
  "_id": ObjectId("..."),
  "alert_id": "ALT-2026-02-06-001",
  "type": "pollution_spike",
  "severity": "high",  // critical, high, medium, low
  "title": "Air Quality Deteriorating",
  "description": "PM2.5 levels predicted to exceed 200 μg/m³ in next 6 hours",
  "location": {
    "type": "Point",
    "coordinates": [72.5714, 23.0225]
  },
  "affected_area": {
    "type": "Polygon",
    "coordinates": [[...]]
  },
  "metrics": {
    "current_aqi": 180,
    "predicted_aqi": 220,
    "confidence": 0.87
  },
  "recommendations": [
    "Implement odd-even vehicle scheme",
    "Issue health advisory",
    "Increase public transport frequency"
  ],
  "status": "active",  // active, acknowledged, resolved
  "created_at": ISODate("2026-02-06T08:00:00Z"),
  "acknowledged_at": null,
  "acknowledged_by": null,
  "resolved_at": null,
  "actions_taken": [],
  "metadata": {
    "model_version": "pollution-xgb-v1.5",
    "data_sources": ["station_001", "station_002"]
  }
}

// Indexes
db.alerts.createIndex({ "created_at": -1 })
db.alerts.createIndex({ "status": 1, "severity": 1 })
db.alerts.createIndex({ "location": "2dsphere" })
db.alerts.createIndex({ "type": 1, "created_at": -1 })
```

**Simulation Results Collection**:
```javascript
{
  "_id": ObjectId("..."),
  "simulation_id": "SIM-2026-02-06-001",
  "scenario": {
    "name": "odd_even_scheme",
    "description": "Simulate impact of odd-even vehicle restriction",
    "parameters": {
      "start_date": "2026-03-01",
      "duration_days": 30,
      "restriction_hours": "08:00-20:00",
      "exemptions": ["emergency", "public_transport"]
    }
  },
  "configuration": {
    "num_agents": 15000,
    "time_step_hours": 1,
    "monte_carlo_runs": 1000
  },
  "results": {
    "traffic": {
      "avg_congestion_reduction": 28.5,  // percentage
      "peak_hour_improvement": 35.2,
      "confidence_interval": [25.1, 31.9]
    },
    "pollution": {
      "avg_aqi_reduction": 22.3,
      "good_air_days_increase": 12,
      "confidence_interval": [18.5, 26.1]
    },
    "economic": {
      "implementation_cost": 5000000,  // INR
      "time_savings_value": 12000000,
      "health_benefits": 8000000,
      "net_benefit": 15000000
    },
    "citizen_impact": {
      "satisfaction_change": -5.2,  // slight decrease
      "compliance_rate": 78.5
    }
  },
  "comparison_with_baseline": {
    "traffic_improvement": 28.5,
    "pollution_improvement": 22.3,
    "cost_effectiveness": 3.0  // benefit per rupee
  },
  "status": "completed",
  "created_at": ISODate("2026-02-06T09:00:00Z"),
  "completed_at": ISODate("2026-02-06T09:15:00Z"),
  "execution_time_seconds": 892,
  "created_by": "user_rajesh_kumar"
}
```

#### 3.1.3 Object Storage (S3)

**Bucket Structure**:
```
digital-twin-data/
├── raw-data/
│   ├── traffic/
│   │   ├── 2026/02/06/
│   │   │   ├── camera-feeds/
│   │   │   │   ├── CAM001_20260206_080000.mp4
│   │   │   │   └── CAM002_20260206_080000.mp4
│   │   │   └── sensor-dumps/
│   │   │       └── traffic_20260206.csv
│   ├── satellite/
│   │   ├── sentinel2/
│   │   │   └── 2026/02/01/
│   │   │       └── S2A_MSIL2A_20260201_T43QGD.zip
│   └── weather/
│       └── 2026/02/
│           └── imd_forecast_20260206.json
├── processed-data/
│   ├── features/
│   │   ├── traffic_features_20260206.parquet
│   │   └── pollution_features_20260206.parquet
│   └── aggregations/
│       └── daily_summaries_202602.parquet
├── models/
│   ├── traffic/
│   │   ├── lstm-v2.3/
│   │   │   ├── model.pth
│   │   │   ├── config.json
│   │   │   └── metrics.json
│   │   └── lstm-v2.4/
│   ├── pollution/
│   │   └── xgboost-v1.5/
│   └── water/
│       └── prophet-v1.2/
├── reports/
│   ├── daily/
│   │   └── report_20260206.pdf
│   └── monthly/
│       └── report_202601.pdf
└── backups/
    ├── database/
    │   └── timescaledb_20260206_0300.dump
    └── mongodb/
        └── mongodb_20260206_0300.archive
```

**Lifecycle Policies**:
```json
{
  "Rules": [
    {
      "Id": "MoveRawDataToIA",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "STANDARD_IA"
        },
        {
          "Days": 90,
          "StorageClass": "GLACIER"
        }
      ]
    },
    {
      "Id": "DeleteOldBackups",
      "Status": "Enabled",
      "Expiration": {
        "Days": 90
      },
      "Filter": {
        "Prefix": "backups/"
      }
    }
  ]
}
```

### 3.2 Data Flow

**Real-Time Data Flow**:
```
Sensors → IoT Gateway → Kafka → Flink → TimescaleDB → API → Dashboard
                                    ↓
                                  Redis (Cache)
                                    ↓
                                ML Models
```

**Batch Data Flow**:
```
External APIs → S3 → Spark → Feature Store → ML Training → Model Registry
                                                              ↓
                                                        Model Serving
```

**Prediction Flow**:
```
User Request → API Gateway → ML Service → Feature Store → Model Inference
                                              ↓
                                          TimescaleDB (Historical)
                                              ↓
                                          Redis (Recent)
                                              ↓
                                          Response
```


---

## 4. ML Model Architecture

### 4.1 Model Training Pipeline

**Architecture**:

```
┌─────────────────────────────────────────────────────────┐
│              ML Training Pipeline                        │
│                                                          │
│  ┌──────────────────────────────────────────────┐       │
│  │  Data Preparation (Apache Spark)             │       │
│  │  - Feature extraction                        │       │
│  │  - Data cleaning & validation                │       │
│  │  - Train/validation/test split               │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Feature Engineering                         │       │
│  │  - Lag features (1h, 24h, 7d)                │       │
│  │  - Rolling statistics (mean, std, min, max)  │       │
│  │  - Cyclical encoding (hour, day, month)      │       │
│  │  - Interaction features                      │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Model Training                              │       │
│  │  ┌────────────────────────────────────────┐  │       │
│  │  │  Hyperparameter Tuning (Optuna)       │  │       │
│  │  │  - Bayesian optimization              │  │       │
│  │  │  - 100+ trials                        │  │       │
│  │  └────────────────────────────────────────┘  │       │
│  │  ┌────────────────────────────────────────┐  │       │
│  │  │  Cross-Validation                     │  │       │
│  │  │  - Time-series split (5 folds)        │  │       │
│  │  │  - Walk-forward validation            │  │       │
│  │  └────────────────────────────────────────┘  │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Model Evaluation                            │       │
│  │  - Accuracy metrics (MAPE, RMSE, R²)         │       │
│  │  - Business metrics (cost, impact)           │       │
│  │  - Fairness & bias checks                    │       │
│  │  - Explainability (SHAP, LIME)               │       │
│  └──────────────┬───────────────────────────────┘       │
│                 │                                        │
│                 ▼                                        │
│  ┌──────────────────────────────────────────────┐       │
│  │  Model Registry (MLflow)                     │       │
│  │  - Version control                           │       │
│  │  - Metadata & lineage                        │       │
│  │  - Approval workflow                         │       │
│  └──────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Model Details

#### 4.2.1 Traffic Prediction Model

**Model**: LSTM (Long Short-Term Memory)

**Architecture**:
```python
class TrafficLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(
            input_size=50,      # 50 features
            hidden_size=128,
            num_layers=2,
            dropout=0.2,
            batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(32, 1)  # Predict traffic volume
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm1(x)
        lstm_out, _ = self.lstm2(lstm_out)
        out = self.fc1(lstm_out[:, -1, :])  # Last timestep
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
```

**Training Configuration**:
```yaml
model: traffic_lstm
hyperparameters:
  sequence_length: 24  # 24 hours of history
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  optimizer: Adam
  loss_function: MSE
  
features:
  - historical_traffic (24 lags)
  - day_of_week (one-hot)
  - hour_of_day (cyclical)
  - weather (temperature, rainfall)
  - events (binary)
  - holidays (binary)
  - nearby_traffic (spatial)
  
target:
  - traffic_volume (next 2, 6, 12, 24 hours)
  
validation:
  method: time_series_split
  n_splits: 5
  test_size: 0.2
```

**Performance Metrics**:
- 2h forecast: MAPE 8.5%, R² 0.92
- 6h forecast: MAPE 11.2%, R² 0.89
- 24h forecast: MAPE 14.8%, R² 0.86

#### 4.2.2 Pollution Prediction Model

**Model**: XGBoost Ensemble

**Configuration**:
```python
xgb_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

# Ensemble of 3 models
models = [
    XGBRegressor(**xgb_params),  # PM2.5
    XGBRegressor(**xgb_params),  # PM10
    XGBRegressor(**xgb_params)   # AQI
]
```

**Feature Importance** (Top 10):
1. Historical PM2.5 (24h lag) - 18.5%
2. Wind speed - 12.3%
3. Temperature - 10.8%
4. Traffic volume - 9.2%
5. Humidity - 8.7%
6. Historical PM10 (24h lag) - 7.9%
7. Hour of day - 6.5%
8. Industrial activity index - 5.8%
9. Wind direction - 5.2%
10. Day of week - 4.1%

**Performance Metrics**:
- 24h PM2.5 forecast: RMSE 18.2 μg/m³, R² 0.87
- 48h PM2.5 forecast: RMSE 23.5 μg/m³, R² 0.82
- AQI category accuracy: 88.5%

#### 4.2.3 Water Demand Model

**Model**: Prophet + Neural Prophet Ensemble

**Prophet Configuration**:
```python
from prophet import Prophet

model = Prophet(
    growth='linear',
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05
)

# Add custom seasonalities
model.add_seasonality(
    name='monthly',
    period=30.5,
    fourier_order=5
)

# Add regressors
model.add_regressor('temperature')
model.add_regressor('rainfall')
model.add_regressor('holiday')
```

**Performance Metrics**:
- 1-day forecast: MAPE 6.8%, R² 0.94
- 3-day forecast: MAPE 10.2%, R² 0.90
- 7-day forecast: MAPE 13.5%, R² 0.87

### 4.3 Model Monitoring & Retraining

**Monitoring Metrics**:
```python
class ModelMonitor:
    def __init__(self, model_name):
        self.model_name = model_name
        self.metrics = []
    
    def log_prediction(self, prediction, actual, features):
        # Calculate metrics
        error = abs(prediction - actual)
        mape = error / actual * 100
        
        # Feature drift detection
        feature_drift = self.detect_drift(features)
        
        # Prediction drift detection
        prediction_drift = self.detect_prediction_drift(prediction)
        
        # Log to monitoring system
        self.metrics.append({
            'timestamp': datetime.now(),
            'error': error,
            'mape': mape,
            'feature_drift': feature_drift,
            'prediction_drift': prediction_drift
        })
        
        # Alert if drift detected
        if feature_drift > 0.1 or prediction_drift > 0.15:
            self.trigger_retraining_alert()
```

**Retraining Triggers**:
1. **Scheduled**: Weekly for high-frequency models
2. **Performance Degradation**: MAPE increases by >10%
3. **Data Drift**: Feature distribution shifts significantly
4. **Concept Drift**: Relationship between features and target changes
5. **Manual**: Domain expert requests retraining

---

## 5. API Design

### 5.1 RESTful API Endpoints

**Base URL**: `https://api.digitaltwin.gov.in/v1`

#### 5.1.1 Data API

**GET /data/traffic**
```http
GET /data/traffic?location=R001&start=2026-02-06T00:00:00Z&end=2026-02-06T23:59:59Z

Response 200 OK:
{
  "data": [
    {
      "timestamp": "2026-02-06T08:00:00Z",
      "location": {
        "road_id": "R001",
        "coordinates": [23.0225, 72.5714]
      },
      "metrics": {
        "vehicle_count": 450,
        "average_speed": 35.5,
        "congestion_level": 65
      }
    }
  ],
  "pagination": {
    "total": 1440,
    "page": 1,
    "per_page": 100
  }
}
```

**GET /data/pollution**
```http
GET /data/pollution?station=P001&parameter=PM2.5&hours=24

Response 200 OK:
{
  "station_id": "P001",
  "location": [23.0225, 72.5714],
  "parameter": "PM2.5",
  "unit": "μg/m³",
  "data": [
    {
      "timestamp": "2026-02-06T08:00:00Z",
      "value": 85.5,
      "aqi": 165,
      "category": "Moderate"
    }
  ],
  "statistics": {
    "min": 45.2,
    "max": 125.8,
    "mean": 82.3,
    "std": 18.5
  }
}
```

#### 5.1.2 Prediction API

**POST /predict/traffic**
```http
POST /predict/traffic
Content-Type: application/json

{
  "location": {
    "road_id": "R001",
    "coordinates": [23.0225, 72.5714]
  },
  "forecast_horizon": "24h",
  "include_confidence": true,
  "include_explanation": true
}

Response 200 OK:
{
  "request_id": "pred-20260206-001",
  "model_version": "traffic-lstm-v2.3",
  "predictions": [
    {
      "timestamp": "2026-02-06T10:00:00Z",
      "traffic_volume": 450,
      "average_speed": 35.5,
      "congestion_level": 65,
      "confidence_interval": {
        "lower": 400,
        "upper": 500,
        "confidence": 0.95
      }
    }
  ],
  "explanation": {
    "top_features": [
      {"feature": "historical_traffic_24h", "importance": 0.25},
      {"feature": "hour_of_day", "importance": 0.18},
      {"feature": "day_of_week", "importance": 0.15}
    ]
  },
  "inference_time_ms": 245
}
```

**POST /predict/pollution**
```http
POST /predict/pollution
Content-Type: application/json

{
  "station_id": "P001",
  "parameters": ["PM2.5", "PM10", "AQI"],
  "forecast_horizon": "48h",
  "include_health_advisory": true
}

Response 200 OK:
{
  "request_id": "pred-20260206-002",
  "model_version": "pollution-xgb-v1.5",
  "predictions": {
    "PM2.5": [
      {
        "timestamp": "2026-02-06T10:00:00Z",
        "value": 95.5,
        "confidence_interval": [85.2, 105.8]
      }
    ],
    "AQI": [
      {
        "timestamp": "2026-02-06T10:00:00Z",
        "value": 175,
        "category": "Moderate",
        "health_advisory": "Sensitive groups should limit outdoor activity"
      }
    ]
  }
}
```

#### 5.1.3 Simulation API

**POST /simulate/scenario**
```http
POST /simulate/scenario
Content-Type: application/json

{
  "scenario_name": "odd_even_vehicle_scheme",
  "description": "Test impact of odd-even vehicle restriction",
  "parameters": {
    "start_date": "2026-03-01",
    "duration_days": 30,
    "restriction_hours": "08:00-20:00",
    "affected_zones": ["Z001", "Z002", "Z003"],
    "exemptions": ["emergency", "public_transport", "electric"]
  },
  "simulation_config": {
    "num_agents": 15000,
    "time_step_hours": 1,
    "monte_carlo_runs": 1000
  }
}

Response 202 Accepted:
{
  "simulation_id": "SIM-2026-02-06-001",
  "status": "queued",
  "estimated_completion_time": "2026-02-06T09:15:00Z",
  "status_url": "/simulate/results/SIM-2026-02-06-001"
}
```

**GET /simulate/results/{simulation_id}**
```http
GET /simulate/results/SIM-2026-02-06-001

Response 200 OK:
{
  "simulation_id": "SIM-2026-02-06-001",
  "status": "completed",
  "scenario": {...},
  "results": {
    "traffic": {
      "avg_congestion_reduction": 28.5,
      "peak_hour_improvement": 35.2,
      "confidence_interval": [25.1, 31.9]
    },
    "pollution": {
      "avg_aqi_reduction": 22.3,
      "good_air_days_increase": 12
    },
    "economic": {
      "implementation_cost": 5000000,
      "time_savings_value": 12000000,
      "net_benefit": 15000000
    }
  },
  "comparison_with_baseline": {
    "traffic_improvement": 28.5,
    "pollution_improvement": 22.3,
    "cost_effectiveness": 3.0
  },
  "execution_time_seconds": 892
}
```

#### 5.1.4 Recommendation API

**GET /recommend/interventions**
```http
GET /recommend/interventions?context=high_pollution&budget=10000000

Response 200 OK:
{
  "context": {
    "current_aqi": 220,
    "trend": "increasing",
    "budget_available": 10000000
  },
  "recommendations": [
    {
      "rank": 1,
      "intervention": "odd_even_vehicle_scheme",
      "category": "traffic_pollution",
      "estimated_impact": {
        "aqi_reduction": 25,
        "implementation_time_days": 7,
        "duration_days": 30
      },
      "cost": {
        "implementation": 2000000,
        "operational_per_day": 50000
      },
      "cost_effectiveness": 4.5,
      "confidence": 0.87,
      "justification": "Historical data shows 20-30% AQI reduction during similar interventions",
      "risks": ["Citizen resistance", "Compliance challenges"],
      "similar_cases": [
        {
          "location": "Delhi",
          "date": "2024-11-15",
          "outcome": "28% AQI reduction"
        }
      ]
    }
  ]
}
```

### 5.2 WebSocket API (Real-Time Updates)

**Connection**: `wss://api.digitaltwin.gov.in/v1/ws`

**Subscribe to Traffic Updates**:
```javascript
// Client sends
{
  "action": "subscribe",
  "channel": "traffic",
  "filters": {
    "roads": ["R001", "R002"],
    "update_frequency": "30s"
  }
}

// Server sends updates
{
  "channel": "traffic",
  "timestamp": "2026-02-06T08:00:30Z",
  "data": {
    "road_id": "R001",
    "congestion_level": 68,
    "change_from_previous": +3
  }
}
```

### 5.3 API Security

**Authentication**:
```http
Authorization: Bearer <JWT_TOKEN>
X-API-Key: <API_KEY>
```

**Rate Limiting**:
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1675680000
```

**Error Responses**:
```json
{
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Invalid forecast_horizon. Must be one of: 2h, 6h, 12h, 24h",
    "details": {
      "parameter": "forecast_horizon",
      "provided": "48h",
      "allowed": ["2h", "6h", "12h", "24h"]
    },
    "request_id": "req-20260206-001",
    "timestamp": "2026-02-06T08:00:00Z"
  }
}
```

