from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    dataset_source = Column(String(20), nullable=False)
    flow_ref = Column(String(64), nullable=True)
    src_ip = Column(String(45), nullable=True)
    dst_ip = Column(String(45), nullable=True)
    predicted_attack_cat = Column(String(64), nullable=False)
    severity = Column(String(20), nullable=False)
    risk_score = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=False)
    detector_scores = Column(JSONB, nullable=True)
    threat_intel_match = Column(JSONB, nullable=True)
    status = Column(String(20), nullable=False, default="new")
    raw_features = Column(JSONB, nullable=True)


class PredictionLog(Base):
    __tablename__ = "predictions_log"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    dataset_source = Column(String(20), nullable=False)
    predicted_attack_cat = Column(String(64), nullable=False)
    binary_flag = Column(Integer, nullable=False)  # 1 if the detection gate flagged this flow, else 0
    anomaly_score = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    risk_score = Column(Float, nullable=True)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(Integer, primary_key=True)
    model_name = Column(String(64), nullable=False)
    dataset = Column(String(20), nullable=False)
    version = Column(String(32), nullable=False)
    trained_at = Column(DateTime(timezone=True), nullable=False)
    metrics = Column(JSONB, nullable=True)
    artifact_path = Column(String(255), nullable=False)
