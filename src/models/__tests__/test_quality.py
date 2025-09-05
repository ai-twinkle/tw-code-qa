"""
Tests for quality.py module

Tests all quality assessment related models and their methods
to ensure comprehensive coverage.
"""

import pytest
import time
from typing import Dict
from src.models.quality import (
    ErrorType, 
    ErrorRecord, 
    QualityMetric, 
    QualityAssessment, 
    QualityReport, 
    BatchQualityReport
)


class TestErrorRecord:
    """Test ErrorRecord model"""
    
    def test_error_record_creation_valid(self):
        """Test valid ErrorRecord creation"""
        error = ErrorRecord(
            error_type=ErrorType.API_CONNECTION,
            error_message="Connection failed",
            timestamp=time.time(),
            retry_attempt=1,
            agent_name="test_agent",
            recovery_action="retry",
            context_data={"key": "value"}
        )
        
        assert error.error_type == ErrorType.API_CONNECTION
        assert error.error_message == "Connection failed"
        assert error.retry_attempt == 1
        assert error.agent_name == "test_agent"
        assert error.recovery_action == "retry"
        assert error.context_data == {"key": "value"}
    
    def test_error_record_empty_message_validation(self):
        """Test ErrorRecord validation with empty message - covers line 39"""
        with pytest.raises(ValueError, match="Error message cannot be empty"):
            ErrorRecord(
                error_type=ErrorType.API_CONNECTION,
                error_message="",  # Empty message should raise ValueError
                timestamp=time.time(),
                retry_attempt=1,
                agent_name="test_agent",
                recovery_action="retry"
            )
    
    def test_error_record_all_error_types(self):
        """Test ErrorRecord with all ErrorType values"""
        for error_type in ErrorType:
            error = ErrorRecord(
                error_type=error_type,
                error_message="Test message",
                timestamp=time.time(),
                retry_attempt=1,
                agent_name="test_agent",
                recovery_action="retry"
            )
            assert error.error_type == error_type


class TestQualityMetric:
    """Test QualityMetric model"""
    
    def test_quality_metric_creation_valid(self):
        """Test valid QualityMetric creation"""
        metric = QualityMetric(
            metric_name="test_metric",
            score=7.5,
            max_score=10.0,
            description="Test metric description"
        )
        
        assert metric.metric_name == "test_metric"
        assert metric.score == 7.5
        assert metric.max_score == 10.0
        assert metric.description == "Test metric description"
    
    def test_quality_metric_score_validation_negative(self):
        """Test QualityMetric validation with negative score - covers lines 52-53"""
        with pytest.raises(ValueError, match="Score -1.0 must be between 0 and 10.0"):
            QualityMetric(
                metric_name="test_metric",
                score=-1.0,  # Negative score should raise ValueError
                max_score=10.0,
                description="Test metric description"
            )
    
    def test_quality_metric_score_validation_exceeds_max(self):
        """Test QualityMetric validation with score exceeding max - covers lines 52-53"""
        with pytest.raises(ValueError, match="Score 15.0 must be between 0 and 10.0"):
            QualityMetric(
                metric_name="test_metric",
                score=15.0,  # Score exceeding max should raise ValueError
                max_score=10.0,
                description="Test metric description"
            )
    
    def test_quality_metric_get_percentage(self):
        """Test QualityMetric get_percentage method - covers line 57"""
        metric = QualityMetric(
            metric_name="test_metric",
            score=7.5,
            max_score=10.0,
            description="Test metric description"
        )
        
        percentage = metric.get_percentage()
        assert percentage == 75.0
        
        # Test with different values
        metric_half = QualityMetric(
            metric_name="half_metric",
            score=5.0,
            max_score=10.0,
            description="Half score metric"
        )
        assert metric_half.get_percentage() == 50.0


class TestQualityAssessment:
    """Test QualityAssessment model"""
    
    def test_quality_assessment_creation_with_auto_calculation(self):
        """Test QualityAssessment with automatic overall score calculation - covers line 84"""
        assessment = QualityAssessment(
            record_id="test_001",
            semantic_consistency_score=8.0,
            code_integrity_score=7.0,
            translation_naturalness_score=9.0,
            overall_quality_score=0,  # Set to 0 to trigger auto calculation
            semantic_analysis="Good semantic consistency",
            code_analysis="Minor code issues",
            naturalness_analysis="Very natural translation",
            improvement_suggestions=["Improve variable naming"],
            evaluator_model="gpt-4o"
        )
        
        # Check auto-calculated overall score: 8.0*0.5 + 7.0*0.3 + 9.0*0.2 = 4.0 + 2.1 + 1.8 = 7.9
        expected_score = 8.0 * 0.5 + 7.0 * 0.3 + 9.0 * 0.2
        assert assessment.overall_quality_score == expected_score
    
    def test_quality_assessment_creation_with_manual_score(self):
        """Test QualityAssessment with manual overall score (no auto calculation)"""
        manual_score = 8.5
        assessment = QualityAssessment(
            record_id="test_002",
            semantic_consistency_score=8.0,
            code_integrity_score=7.0,
            translation_naturalness_score=9.0,
            overall_quality_score=manual_score,  # Manual score, should not be overridden
            semantic_analysis="Good semantic consistency",
            code_analysis="Minor code issues",
            naturalness_analysis="Very natural translation"
        )
        
        assert assessment.overall_quality_score == manual_score
    
    def test_quality_assessment_is_acceptable_quality_true(self):
        """Test is_acceptable_quality returns True for score >= 7.0 - covers line 92"""
        assessment = QualityAssessment(
            record_id="test_003",
            semantic_consistency_score=8.0,
            code_integrity_score=8.0,
            translation_naturalness_score=8.0,
            overall_quality_score=8.0,
            semantic_analysis="Good",
            code_analysis="Good",
            naturalness_analysis="Good"
        )
        
        assert assessment.is_acceptable_quality() is True
    
    def test_quality_assessment_is_acceptable_quality_false(self):
        """Test is_acceptable_quality returns False for score < 7.0 - covers line 92"""
        assessment = QualityAssessment(
            record_id="test_004",
            semantic_consistency_score=6.0,
            code_integrity_score=6.0,
            translation_naturalness_score=6.0,
            overall_quality_score=6.0,
            semantic_analysis="Poor",
            code_analysis="Poor",
            naturalness_analysis="Poor"
        )
        
        assert assessment.is_acceptable_quality() is False
    
    def test_quality_assessment_needs_retry_true(self):
        """Test needs_retry returns True for score < 7.0 - covers line 96"""
        assessment = QualityAssessment(
            record_id="test_005",
            semantic_consistency_score=5.0,
            code_integrity_score=5.0,
            translation_naturalness_score=5.0,
            overall_quality_score=5.0,
            semantic_analysis="Poor",
            code_analysis="Poor",
            naturalness_analysis="Poor"
        )
        
        assert assessment.needs_retry() is True
    
    def test_quality_assessment_needs_retry_false(self):
        """Test needs_retry returns False for score >= 7.0 - covers line 96"""
        assessment = QualityAssessment(
            record_id="test_006",
            semantic_consistency_score=8.0,
            code_integrity_score=8.0,
            translation_naturalness_score=8.0,
            overall_quality_score=8.0,
            semantic_analysis="Good",
            code_analysis="Good",
            naturalness_analysis="Good"
        )
        
        assert assessment.needs_retry() is False


class TestQualityReport:
    """Test QualityReport model"""
    
    def test_quality_report_creation(self):
        """Test QualityReport creation"""
        assessment = QualityAssessment(
            record_id="test_007",
            semantic_consistency_score=8.0,
            code_integrity_score=8.0,
            translation_naturalness_score=8.0,
            overall_quality_score=8.0,
            semantic_analysis="Good",
            code_analysis="Good",
            naturalness_analysis="Good"
        )
        
        report = QualityReport(
            record_id="test_007",
            quality_assessment=assessment,
            processing_time=1.5,
            retry_count=0,
            final_status="passed"
        )
        
        assert report.record_id == "test_007"
        assert report.quality_assessment == assessment
        assert report.processing_time == 1.5
        assert report.retry_count == 0
        assert report.final_status == "passed"
        assert len(report.error_history) == 0
    
    def test_quality_report_add_error(self):
        """Test QualityReport add_error method - covers line 111"""
        assessment = QualityAssessment(
            record_id="test_008",
            semantic_consistency_score=6.0,
            code_integrity_score=6.0,
            translation_naturalness_score=6.0,
            overall_quality_score=6.0,
            semantic_analysis="Poor",
            code_analysis="Poor",
            naturalness_analysis="Poor"
        )
        
        report = QualityReport(
            record_id="test_008",
            quality_assessment=assessment,
            processing_time=2.0,
            retry_count=1,
            final_status="failed"
        )
        
        error = ErrorRecord(
            error_type=ErrorType.TRANSLATION_QUALITY,
            error_message="Poor translation quality",
            timestamp=time.time(),
            retry_attempt=1,
            agent_name="evaluator",
            recovery_action="retry"
        )
        
        report.add_error(error)
        
        assert len(report.error_history) == 1
        assert report.error_history[0] == error


class TestBatchQualityReport:
    """Test BatchQualityReport model"""
    
    def test_batch_quality_report_creation(self):
        """Test BatchQualityReport creation"""
        batch_report = BatchQualityReport(
            batch_id="batch_001",
            total_records=10,
            processed_records=8,
            passed_records=6,
            failed_records=2,
            retry_records=2,
            average_quality_score=7.5,
            min_quality_score=5.0,
            max_quality_score=9.0,
            total_processing_time=15.0,
            average_processing_time=1.875,
            total_retries=3
        )
        
        assert batch_report.batch_id == "batch_001"
        assert batch_report.total_records == 10
        assert batch_report.processed_records == 8
        assert batch_report.passed_records == 6
        assert batch_report.failed_records == 2
        assert batch_report.retry_records == 2
    
    def test_batch_quality_report_calculate_statistics_empty(self):
        """Test calculate_statistics with empty reports - covers lines 146-171"""
        batch_report = BatchQualityReport(
            batch_id="batch_002",
            total_records=0,
            processed_records=0,
            passed_records=0,
            failed_records=0,
            retry_records=0,
            average_quality_score=0.0,
            min_quality_score=0.0,
            max_quality_score=0.0,
            total_processing_time=0.0,
            average_processing_time=0.0,
            total_retries=0
        )
        
        # This should return early without processing
        batch_report.calculate_statistics()
        
        # Values should remain as initialized
        assert batch_report.average_quality_score == 0.0
        assert batch_report.min_quality_score == 0.0
        assert batch_report.max_quality_score == 0.0
    
    def test_batch_quality_report_calculate_statistics_with_reports(self):
        """Test calculate_statistics with actual reports - covers lines 146-171"""
        # Create some sample reports
        assessment1 = QualityAssessment(
            record_id="test_009",
            semantic_consistency_score=8.0,
            code_integrity_score=7.0,
            translation_naturalness_score=9.0,
            overall_quality_score=8.0,
            semantic_analysis="Good",
            code_analysis="Good",
            naturalness_analysis="Excellent"
        )
        
        assessment2 = QualityAssessment(
            record_id="test_010",
            semantic_consistency_score=6.0,
            code_integrity_score=6.0,
            translation_naturalness_score=6.0,
            overall_quality_score=6.0,
            semantic_analysis="Fair",
            code_analysis="Fair",
            naturalness_analysis="Fair"
        )
        
        error1 = ErrorRecord(
            error_type=ErrorType.API_CONNECTION,
            error_message="Connection timeout",
            timestamp=time.time(),
            retry_attempt=1,
            agent_name="agent1",
            recovery_action="retry"
        )
        
        error2 = ErrorRecord(
            error_type=ErrorType.TRANSLATION_QUALITY,
            error_message="Poor quality",
            timestamp=time.time(),
            retry_attempt=1,
            agent_name="agent2",
            recovery_action="regenerate"
        )
        
        report1 = QualityReport(
            record_id="test_009",
            quality_assessment=assessment1,
            processing_time=1.5,
            retry_count=1,
            final_status="passed",
            error_history=[error1]
        )
        
        report2 = QualityReport(
            record_id="test_010",
            quality_assessment=assessment2,
            processing_time=2.5,
            retry_count=2,
            final_status="failed",
            error_history=[error2]
        )
        
        batch_report = BatchQualityReport(
            batch_id="batch_003",
            total_records=2,
            processed_records=2,
            passed_records=1,
            failed_records=1,
            retry_records=1,
            average_quality_score=0.0,  # Will be calculated
            min_quality_score=0.0,      # Will be calculated
            max_quality_score=0.0,      # Will be calculated
            total_processing_time=4.0,
            average_processing_time=0.0, # Will be calculated
            total_retries=0,             # Will be calculated
            individual_reports=[report1, report2]
        )
        
        batch_report.calculate_statistics()
        
        # Check calculated values
        assert batch_report.average_quality_score == 7.0  # (8.0 + 6.0) / 2
        assert batch_report.min_quality_score == 6.0
        assert batch_report.max_quality_score == 8.0
        assert batch_report.average_processing_time == 2.0  # (1.5 + 2.5) / 2
        assert batch_report.total_retries == 3  # 1 + 2
        
        # Check error summary
        assert batch_report.error_summary[ErrorType.API_CONNECTION] == 1
        assert batch_report.error_summary[ErrorType.TRANSLATION_QUALITY] == 1
    
    def test_batch_quality_report_get_success_rate_with_records(self):
        """Test get_success_rate with records - covers lines 175-177"""
        batch_report = BatchQualityReport(
            batch_id="batch_004",
            total_records=10,
            processed_records=8,
            passed_records=6,
            failed_records=2,
            retry_records=2,
            average_quality_score=7.5,
            min_quality_score=5.0,
            max_quality_score=9.0,
            total_processing_time=15.0,
            average_processing_time=1.875,
            total_retries=3
        )
        
        success_rate = batch_report.get_success_rate()
        assert success_rate == 0.6  # 6 / 10
    
    def test_batch_quality_report_get_success_rate_zero_records(self):
        """Test get_success_rate with zero records - covers lines 175-177"""
        batch_report = BatchQualityReport(
            batch_id="batch_005",
            total_records=0,
            processed_records=0,
            passed_records=0,
            failed_records=0,
            retry_records=0,
            average_quality_score=0.0,
            min_quality_score=0.0,
            max_quality_score=0.0,
            total_processing_time=0.0,
            average_processing_time=0.0,
            total_retries=0
        )
        
        success_rate = batch_report.get_success_rate()
        assert success_rate == 0.0
    
    def test_batch_quality_report_get_failure_rate_with_records(self):
        """Test get_failure_rate with records - covers lines 181-183"""
        batch_report = BatchQualityReport(
            batch_id="batch_006",
            total_records=10,
            processed_records=8,
            passed_records=6,
            failed_records=4,
            retry_records=2,
            average_quality_score=7.5,
            min_quality_score=5.0,
            max_quality_score=9.0,
            total_processing_time=15.0,
            average_processing_time=1.875,
            total_retries=3
        )
        
        failure_rate = batch_report.get_failure_rate()
        assert failure_rate == 0.4  # 4 / 10
    
    def test_batch_quality_report_get_failure_rate_zero_records(self):
        """Test get_failure_rate with zero records - covers lines 181-183"""
        batch_report = BatchQualityReport(
            batch_id="batch_007",
            total_records=0,
            processed_records=0,
            passed_records=0,
            failed_records=0,
            retry_records=0,
            average_quality_score=0.0,
            min_quality_score=0.0,
            max_quality_score=0.0,
            total_processing_time=0.0,
            average_processing_time=0.0,
            total_retries=0
        )
        
        failure_rate = batch_report.get_failure_rate()
        assert failure_rate == 0.0


class TestQualityModelIntegration:
    """Integration tests for quality models"""
    
    def test_complete_quality_workflow(self):
        """Test a complete quality assessment workflow"""
        # Create error records
        error1 = ErrorRecord(
            error_type=ErrorType.API_CONNECTION,
            error_message="Connection timeout",
            timestamp=time.time(),
            retry_attempt=1,
            agent_name="reproducer",
            recovery_action="retry"
        )
        
        error2 = ErrorRecord(
            error_type=ErrorType.TRANSLATION_QUALITY,
            error_message="Poor translation quality detected",
            timestamp=time.time(),
            retry_attempt=2,
            agent_name="evaluator",
            recovery_action="regenerate"
        )
        
        # Create quality assessments
        assessment1 = QualityAssessment(
            record_id="workflow_001",
            semantic_consistency_score=8.5,
            code_integrity_score=7.5,
            translation_naturalness_score=9.0,
            overall_quality_score=0,  # Auto-calculate
            semantic_analysis="Excellent semantic preservation",
            code_analysis="Minor formatting issues",
            naturalness_analysis="Very natural Chinese translation",
            improvement_suggestions=["Fix indentation", "Add comments"],
            evaluator_model="gpt-4o"
        )
        
        assessment2 = QualityAssessment(
            record_id="workflow_002",
            semantic_consistency_score=6.0,
            code_integrity_score=5.5,
            translation_naturalness_score=6.5,
            overall_quality_score=0,  # Auto-calculate
            semantic_analysis="Some semantic issues",
            code_analysis="Code structure problems",
            naturalness_analysis="Translation needs improvement",
            improvement_suggestions=["Improve code structure", "Better translation"],
            evaluator_model="gpt-4o"
        )
        
        # Create quality reports
        report1 = QualityReport(
            record_id="workflow_001",
            quality_assessment=assessment1,
            processing_time=2.5,
            retry_count=1,
            final_status="passed"
        )
        report1.add_error(error1)
        
        report2 = QualityReport(
            record_id="workflow_002",
            quality_assessment=assessment2,
            processing_time=3.5,
            retry_count=2,
            final_status="needs_manual_review"
        )
        report2.add_error(error2)
        
        # Create batch report
        batch_report = BatchQualityReport(
            batch_id="workflow_batch_001",
            total_records=2,
            processed_records=2,
            passed_records=1,
            failed_records=0,
            retry_records=1,
            average_quality_score=0.0,
            min_quality_score=0.0,
            max_quality_score=0.0,
            total_processing_time=6.0,
            average_processing_time=0.0,
            total_retries=0,
            individual_reports=[report1, report2]
        )
        
        batch_report.calculate_statistics()
        
        # Verify the complete workflow
        assert assessment1.is_acceptable_quality() is True
        assert assessment2.is_acceptable_quality() is False
        assert assessment1.needs_retry() is False
        assert assessment2.needs_retry() is True
        
        assert len(report1.error_history) == 1
        assert len(report2.error_history) == 1
        
        assert batch_report.get_success_rate() == 0.5  # 1 / 2
        assert batch_report.get_failure_rate() == 0.0  # 0 / 2
        assert batch_report.total_retries == 3  # 1 + 2
        
        # Verify error statistics
        assert ErrorType.API_CONNECTION in batch_report.error_summary
        assert ErrorType.TRANSLATION_QUALITY in batch_report.error_summary
        assert batch_report.error_summary[ErrorType.API_CONNECTION] == 1
        assert batch_report.error_summary[ErrorType.TRANSLATION_QUALITY] == 1
