"""
Data Ingestion Module for Digital Twin
Handles real-time data collection from multiple sources
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List
import asyncio
from kafka import KafkaProducer
import json


@dataclass
class SensorData:
    sensor_id: str
    timestamp: datetime
    data_type: str
    value: float
    unit: str
    location: Dict[str, float]
    metadata: Dict[str, Any]


class DataSource(ABC):
    @abstractmethod
    async def collect(self) -> List[SensorData]:
        pass


class TrafficDataSource(DataSource):
    def __init__(self, sensor_ids: List[str]):
        self.sensor_ids = sensor_ids
    
    async def collect(self) -> List[SensorData]:
        # Simulate traffic data collection
        data = []
        for sensor_id in self.sensor_ids:
            data.append(SensorData(
                sensor_id=sensor_id,
                timestamp=datetime.now(),
                data_type="traffic_flow",
                value=0.0,  # Replace with actual API call
                unit="vehicles/hour",
                location={"lat": 0.0, "lon": 0.0},
                metadata={"road_type": "arterial"}
            ))
        return data


class PollutionDataSource(DataSource):
    def __init__(self, monitor_ids: List[str]):
        self.monitor_ids = monitor_ids
    
    async def collect(self) -> List[SensorData]:
        data = []
        for monitor_id in self.monitor_ids:
            for param in ["PM2.5", "PM10", "CO2", "NOx"]:
                data.append(SensorData(
                    sensor_id=monitor_id,
                    timestamp=datetime.now(),
                    data_type=f"pollution_{param}",
                    value=0.0,
                    unit="μg/m³" if "PM" in param else "ppm",
                    location={"lat": 0.0, "lon": 0.0},
                    metadata={"parameter": param}
                ))
        return data


class WaterDataSource(DataSource):
    def __init__(self, meter_ids: List[str]):
        self.meter_ids = meter_ids
    
    async def collect(self) -> List[SensorData]:
        data = []
        for meter_id in self.meter_ids:
            data.append(SensorData(
                sensor_id=meter_id,
                timestamp=datetime.now(),
                data_type="water_consumption",
                value=0.0,
                unit="liters",
                location={"lat": 0.0, "lon": 0.0},
                metadata={"meter_type": "smart"}
            ))
        return data


class DataIngestionPipeline:
    def __init__(self, kafka_bootstrap_servers: str):
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.sources: List[DataSource] = []
    
    def add_source(self, source: DataSource):
        self.sources.append(source)
    
    async def ingest(self):
        while True:
            for source in self.sources:
                data = await source.collect()
                for sensor_data in data:
                    self._publish(sensor_data)
            await asyncio.sleep(60)
    
    def _publish(self, data: SensorData):
        topic = f"digital-twin.{data.data_type}"
        message = {
            "sensor_id": data.sensor_id,
            "timestamp": data.timestamp.isoformat(),
            "value": data.value,
            "unit": data.unit,
            "location": data.location,
            "metadata": data.metadata
        }
        self.producer.send(topic, value=message)


async def main():
    pipeline = DataIngestionPipeline("localhost:9092")
    
    # Add data sources
    pipeline.add_source(TrafficDataSource(["T001", "T002", "T003"]))
    pipeline.add_source(PollutionDataSource(["P001", "P002"]))
    pipeline.add_source(WaterDataSource(["W001", "W002"]))
    
    await pipeline.ingest()


if __name__ == "__main__":
    asyncio.run(main())
