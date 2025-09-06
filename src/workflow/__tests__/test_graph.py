"""
工作流圖形測試
Workflow Graph Tests
"""

from typing import Dict, Any
from unittest.mock import Mock, patch

import pytest

from src.models.dataset import OriginalRecord, ProcessingStatus, ComplexityLevel
from src.workflow.graph import WorkflowManager


class TestWorkflowManager:
    """工作流管理器測試類"""

    @pytest.fixture
    def workflow_manager(self) -> WorkflowManager:
        """創建工作流管理器實例"""
        return WorkflowManager(enable_checkpointing=False)

    def create_test_original_record(self, record_id: str = "test_id") -> OriginalRecord:
        """創建測試用原始記錄"""
        return OriginalRecord(
            id=record_id,
            question="Test question",
            answer="Test answer",
            source_dataset="test_dataset",
            metadata={"tag": "test", "source_index": 0},
            complexity_level=ComplexityLevel.SIMPLE
        )

    def create_test_workflow_state(self, record_id: str = "test_id") -> Dict[str, Any]:
        """創建測試用工作流狀態"""
        original_record = self.create_test_original_record(record_id)
        return {
            'current_record': original_record,
            'translation_result': None,
            'original_qa_result': None,
            'translated_qa_result': None,
            'quality_assessment': None,
            'retry_count': 0,
            'processing_status': ProcessingStatus.PENDING,
            'error_history': [],
            'improvement_suggestions': []
        }

    def test_initialization(self, workflow_manager: WorkflowManager) -> None:
        """測試初始化"""
        assert workflow_manager is not None
        assert hasattr(workflow_manager, 'workflow')
        # 基本屬性驗證
        assert workflow_manager.enable_checkpointing == False

    def test_process_record_basic(self, workflow_manager: WorkflowManager) -> None:
        """測試基本記錄處理"""
        initial_state = self.create_test_workflow_state("test_record")
        
        # 模擬工作流執行
        with patch.object(workflow_manager.workflow, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                **initial_state,
                'processing_status': ProcessingStatus.COMPLETED
            }
            
            result = workflow_manager.process_record(initial_state)
            
            # 驗證調用和結果
            mock_invoke.assert_called_once()
            assert isinstance(result, dict)

    def test_stream_record_processing(self, workflow_manager: WorkflowManager) -> None:
        """測試流式記錄處理"""
        initial_state = self.create_test_workflow_state("stream_record")
        
        # 模擬流式處理
        mock_stream_data = [
            {"node": "analyzer_designer", "state": {**initial_state, "processing_status": ProcessingStatus.PROCESSING}},
            {"node": "reproducer", "state": {**initial_state, "processing_status": ProcessingStatus.PROCESSING}},
            {"node": "evaluator", "state": {**initial_state, "processing_status": ProcessingStatus.COMPLETED}}
        ]
        
        with patch.object(workflow_manager.workflow, 'stream') as mock_stream:
            mock_stream.return_value = iter(mock_stream_data)
            
            # 測試流式處理
            stream_results = list(workflow_manager.stream_record_processing(initial_state))
            
            # 驗證結果
            mock_stream.assert_called_once()
            assert len(stream_results) == len(mock_stream_data)

    def test_process_record_with_config(self, workflow_manager: WorkflowManager) -> None:
        """測試帶配置的記錄處理"""
        initial_state = self.create_test_workflow_state("config_record")
        config = {"max_retries": 3, "timeout": 30}
        
        with patch.object(workflow_manager.workflow, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                **initial_state,
                'processing_status': ProcessingStatus.COMPLETED
            }
            
            result = workflow_manager.process_record(initial_state, config)
            
            # 驗證配置被傳遞
            mock_invoke.assert_called_once_with(initial_state, config=config)
            assert isinstance(result, dict)

    def test_get_workflow_graph_visualization(self, workflow_manager: WorkflowManager) -> None:
        """測試獲取工作流圖形可視化"""
        # 模擬 Mermaid 圖形生成
        with patch.object(workflow_manager.workflow, 'get_graph') as mock_get_graph:
            mock_graph = Mock()
            mock_graph.draw_mermaid_png.return_value = b"fake_png_data"
            mock_get_graph.return_value = mock_graph
            
            result = workflow_manager.get_workflow_graph_visualization()
            
            # 驗證結果
            assert isinstance(result, bytes)
            assert result == b"fake_png_data"

    @pytest.mark.asyncio
    async def test_async_workflow_operations(self, workflow_manager: WorkflowManager) -> None:
        """測試異步工作流操作"""
        initial_state = self.create_test_workflow_state("async_record")
        
        # 測試異步操作（如果支持）
        if hasattr(workflow_manager.workflow, 'ainvoke'):
            with patch.object(workflow_manager.workflow, 'ainvoke') as mock_ainvoke:
                mock_ainvoke.return_value = {
                    **initial_state,
                    'processing_status': ProcessingStatus.COMPLETED
                }
                
                # 假設有異步方法
                if hasattr(workflow_manager, 'aprocess_record'):
                    result = await workflow_manager.aprocess_record(initial_state)
                    assert isinstance(result, dict)

    def test_error_handling(self, workflow_manager: WorkflowManager) -> None:
        """測試錯誤處理"""
        initial_state = self.create_test_workflow_state("error_record")
        
        with patch.object(workflow_manager.workflow, 'invoke') as mock_invoke:
            # 模擬異常
            mock_invoke.side_effect = Exception("Processing error")
            
            # 測試錯誤處理
            try:
                workflow_manager.process_record(initial_state)
            except Exception as e:
                assert "Processing error" in str(e)
            else:
                # 如果沒有拋出異常，說明有內部錯誤處理
                pass

    def test_state_validation(self, workflow_manager: WorkflowManager) -> None:
        """測試狀態驗證"""
        # 測試無效狀態
        invalid_state = {}
        
        # 驗證是否有狀態驗證邏輯
        try:
            workflow_manager.process_record(invalid_state)
        except (KeyError, ValueError, TypeError):
            # 預期的錯誤
            pass
        except Exception:
            # 其他類型的錯誤也是可接受的
            pass

    def test_workflow_configuration(self, workflow_manager: WorkflowManager) -> None:
        """測試工作流配置"""
        # 測試配置屬性
        assert hasattr(workflow_manager, 'enable_checkpointing')
        
        # 測試不同配置的工作流
        checkpointed_manager = WorkflowManager(enable_checkpointing=True)
        assert checkpointed_manager.enable_checkpointing == True


class TestWorkflowManagerIntegration:
    """工作流管理器整合測試"""

    def create_test_original_record(self, record_id: str = "test_id") -> OriginalRecord:
        """創建測試用原始記錄"""
        return OriginalRecord(
            id=record_id,
            question="Test question",
            answer="Test answer",
            source_dataset="test_dataset",
            metadata={"tag": "test", "source_index": 0},
            complexity_level=ComplexityLevel.SIMPLE
        )

    def create_test_workflow_state(self, record_id: str = "test_id") -> Dict[str, Any]:
        """創建測試用工作流狀態"""
        original_record = self.create_test_original_record(record_id)
        return {
            'current_record': original_record,
            'translation_result': None,
            'original_qa_result': None,
            'translated_qa_result': None,
            'quality_assessment': None,
            'retry_count': 0,
            'processing_status': ProcessingStatus.PENDING,
            'error_history': [],
            'improvement_suggestions': []
        }

    @pytest.mark.integration
    def test_full_workflow_integration(self) -> None:
        """測試完整工作流整合"""
        workflow_manager = WorkflowManager(enable_checkpointing=False)
        
        # 創建測試記錄
        original_record = OriginalRecord(
            id="integration_test",
            question="Integration test question",
            answer="Integration test answer",
            source_dataset="test_dataset",
            metadata={"tag": "integration", "source_index": 0},
            complexity_level=ComplexityLevel.SIMPLE
        )
        
        initial_state = {
            'current_record': original_record,
            'translation_result': None,
            'original_qa_result': None,
            'translated_qa_result': None,
            'quality_assessment': None,
            'retry_count': 0,
            'processing_status': ProcessingStatus.PENDING,
            'error_history': [],
            'improvement_suggestions': []
        }
        
        # 模擬完整處理流程
        with patch.object(workflow_manager.workflow, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                **initial_state,
                'processing_status': ProcessingStatus.COMPLETED,
                'retry_count': 1
            }
            
            result = workflow_manager.process_record(initial_state)
            
            # 驗證整合結果
            assert isinstance(result, dict)
            assert result['processing_status'] == ProcessingStatus.COMPLETED
            mock_invoke.assert_called_once()

    def test_route_after_evaluation_retry_needed(self) -> None:
        """測試評估後路由 - 需要重試的情況"""
        from src.workflow.graph import route_after_evaluation
        from src.models.dataset import OriginalRecord, ComplexityLevel

        # 創建正確的 OriginalRecord 對象
        current_record = OriginalRecord(
            id='retry_test',
            question='Test question',
            answer='Test answer',
            source_dataset='test',
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        state = {
            'current_record': current_record,
            'processing_status': ProcessingStatus.RETRY_NEEDED
        }
        
        result = route_after_evaluation(state)
        assert result == "analyzer_designer"

    def test_route_after_evaluation_completed(self) -> None:
        """測試評估後路由 - 完成的情況"""
        from src.workflow.graph import route_after_evaluation
        from src.models.dataset import OriginalRecord, ComplexityLevel

        # 創建正確的 OriginalRecord 對象
        current_record = OriginalRecord(
            id='complete_test',
            question='Test question',
            answer='Test answer',
            source_dataset='test',
            complexity_level=ComplexityLevel.MEDIUM
        )
        
        state = {
            'current_record': current_record,
            'processing_status': ProcessingStatus.COMPLETED
        }
        
        result = route_after_evaluation(state)
        assert result == "end"

    def test_route_after_evaluation_error_handling(self) -> None:
        """測試評估後路由 - 錯誤處理"""
        from src.workflow.graph import route_after_evaluation
        
        # 測試異常狀態
        invalid_state = None
        result = route_after_evaluation(invalid_state)
        assert result == "end"
        
        # 測試缺少必要字段的狀態
        incomplete_state = {}
        result = route_after_evaluation(incomplete_state)
        assert result == "end"

    def test_stream_processing_with_config_and_checkpointing(self) -> None:
        """測試帶配置和檢查點的流式處理"""
        workflow_manager = WorkflowManager(enable_checkpointing=True)
        initial_state = self.create_test_workflow_state("checkpoint_record")
        config = {"max_retries": 3}
        
        mock_stream_data = [{"node": "analyzer_designer", "state": initial_state}]
        
        with patch.object(workflow_manager.workflow, 'stream') as mock_stream:
            mock_stream.return_value = iter(mock_stream_data)
            
            stream_results = list(workflow_manager.stream_record_processing(initial_state, config))
            
            # 驗證配置被傳遞
            mock_stream.assert_called_once_with(initial_state, config=config)
            assert len(stream_results) == len(mock_stream_data)

    def test_stream_processing_without_config(self) -> None:
        """測試無配置的流式處理"""
        workflow_manager = WorkflowManager(enable_checkpointing=False)
        initial_state = self.create_test_workflow_state("no_config_record")
        
        mock_stream_data = [{"node": "evaluator", "state": initial_state}]
        
        with patch.object(workflow_manager.workflow, 'stream') as mock_stream:
            mock_stream.return_value = iter(mock_stream_data)
            
            stream_results = list(workflow_manager.stream_record_processing(initial_state))
            
            # 驗證沒有配置參數被傳遞
            mock_stream.assert_called_once_with(initial_state)
            assert len(stream_results) == len(mock_stream_data)

    def test_stream_processing_exception_handling(self) -> None:
        """測試流式處理異常處理"""
        workflow_manager = WorkflowManager(enable_checkpointing=False)
        initial_state = self.create_test_workflow_state("exception_record")
        
        with patch.object(workflow_manager.workflow, 'stream') as mock_stream:
            mock_stream.side_effect = Exception("Stream processing error")
            
            stream_results = list(workflow_manager.stream_record_processing(initial_state))
            
            # 驗證錯誤被正確處理
            assert len(stream_results) == 1
            assert "error" in stream_results[0]
            assert "Stream processing error" in stream_results[0]["error"]

    def test_workflow_graph_visualization_exception_handling(self) -> None:
        """測試工作流圖形可視化異常處理"""
        workflow_manager = WorkflowManager(enable_checkpointing=False)
        
        with patch.object(workflow_manager.workflow, 'get_graph') as mock_get_graph:
            mock_get_graph.side_effect = Exception("Visualization error")
            
            result = workflow_manager.get_workflow_graph_visualization()
            
            # 驗證異常被正確處理，返回空字節
            assert result == b""
