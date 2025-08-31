from __future__ import annotations

import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from tensorrt_llm._utils import nvtx_range  # NVIDIA工具，用于性能分析范围标记
from tensorrt_llm.logger import logger

from ..pyexecutor.guided_decoder import GuidedDecoder  # 引导式解码器
from ..pyexecutor.handle_logits import HandleLogits  # 处理logits的工具
from ..pyexecutor.llm_request import LlmRequest, LlmRequestState # 请求定义和状态
from ..pyexecutor.resource_manager import BaseResourceManager, ResourceManager  # 资源管理
from ..pyexecutor.sampler import Sampler, SampleState, TorchSampler  # 采样器
from ..pyexecutor.scheduler import ScheduledRequests  # 请求调度
from ..pyexecutor.seq_slot_manager import SeqSlotManager  # 序列槽管理
from .drafter import Drafter # 草案生成器基类

if TYPE_CHECKING:
    from ..pyexecutor.model_engine import ModelEngine  # 类型检查时导入
    from .interface import SpeculativeDecodingMode  # 推测解码模式接口

# 这段代码实现了一个基于模型的推测解码（Speculative Decoding）草案生成器（Drafter），
# 它使用一个较小的草案模型（draft model）来生成候选令牌（draft tokens），以提高主目标模型的推理效率。

# eagle3 也会用到这个模块的代码

# Place the tool function here to avoid circular import
# 工具函数：获取草案模型的输入提示
def get_draft_model_prompt(spec_dec_mode: SpeculativeDecodingMode,
                           input_tokens: torch.Tensor) -> torch.Tensor:
    """
    Can be used to modify prompts for speculative algorithms that need to update tokens
    before drafting.
    
    用于修改推测解码算法中草案模型的输入提示。
    某些算法（如EAGLE3）需要在 draft model 处理前更新 token。
    """
    if spec_dec_mode.is_eagle3():
        # EAGLE3 always throws away the first token when processing draft inputs
        # EAGLE3总是在处理草案输入时丢弃第一个 token
        return input_tokens[1:]
    return input_tokens


class ModelDrafter(Drafter):
    """Model-based drafter that uses a draft model to generate draft tokens."""
    """基于模型的草案生成器，使用 draft model 生成草案 draft token。"""

    def __init__(
        self,
        spec_config: "DecodingBaseConfig",  # 推测解码配置
        draft_model_engine: "ModelEngine",  # 草案模型引擎
        max_draft_tokens: int,  # 最大草案 token 数
        draft_seq_slot_manager: SeqSlotManager, # 草案序列槽管理器
        sampler: Sampler, # 采样器
        spec_resource_manager: Optional[BaseResourceManager] = None,  # 推测解码资源管理器
        guided_decoder: Optional[GuidedDecoder] = None,  # 引导式解码器
    ):
        super().__init__(spec_config.max_concurrency)  # 初始化父类

        # Validate required parameters
        # 参数验证
        if draft_model_engine is None:
            raise ValueError("draft_model_engine cannot be None")
        if max_draft_tokens < 0:
            raise ValueError(f"max_draft_tokens must be >= 0")

        # Model and resource management
        # 模型和资源管理
        self.draft_model_engine = draft_model_engine
        self.draft_seq_slot_manager = draft_seq_slot_manager
        self.spec_resource_manager = spec_resource_manager

        # Configuration
        # 配置
        self.spec_config = spec_config
        self.max_draft_tokens = max_draft_tokens
        
        # Sampling
        # 采样相关
        self.sampler = sampler
        self._request_draft_logits = False
        if isinstance(sampler, TorchSampler):  # TorchSampler 里有 draft token 接受判断逻辑
            # 如果使用TorchSampler且启用混合采样，需要请求草案logits
            self._request_draft_logits = sampler.enable_mixed_sampler
        self.guided_decoder = guided_decoder

    def _create_draft_request(self, request: LlmRequest,
                              input_tokens: Optional[List]) -> LlmRequest:
        """Create a draft request with common parameters."""
        """创建草案请求的通用参数。"""
        return LlmRequest(input_tokens=input_tokens,    # 输入的 tokens
                          request_id=request.py_request_id, # 使用原始请求ID
                          max_new_tokens=request.py_max_new_tokens,
                          sampling_config=request.sampling_config, # 采样配置
                          guided_decoding_params=request.guided_decoding_params,  # 引导解码参数
                          target_seq_slot=request.py_seq_slot, # 目标序列槽
                          return_perf_metrics=request.return_perf_metrics, # 是否返回性能指标
                          is_streaming=False, # 草案请求非流式
                          is_draft=True,  # 标记为草案请求
                          return_generation_logits=self._request_draft_logits) # 是否返回生成logits

    def _initialize_draft_tokens(self, request: LlmRequest) -> Tuple[int, int]:
        """Initialize draft token tracking for a request."""
        """初始化请求的草案令牌跟踪。"""
        
        # 生成的草案 token 个数
        num_draft_tokens = len(
            request.py_last_draft_tokens
        ) if request.py_last_draft_tokens is not None else 0
        
        request.py_draft_tokens = []  # 清空当前草案令牌

        # 已接受的令牌数
        num_accepted_tokens = request.py_num_accepted_draft_tokens
        # 被拒绝的令牌数
        num_rejected_tokens = num_draft_tokens - num_accepted_tokens
        
        assert num_rejected_tokens >= 0

        return num_draft_tokens, num_accepted_tokens

    def _create_context_request(self, request: LlmRequest,
                                input_tokens: Any) -> LlmRequest:
        """Create a context request for first-time drafting."""
        """为首次草案创建 context 请求。"""
        
        # 请求 draft 模型
        new_request = self._create_draft_request(request, input_tokens)
        
        # 处理分块上下文
        begin_compute, end_compute = request.py_last_context_chunk
        if begin_compute is not None:
            new_request.context_current_position = begin_compute
            new_request.context_chunk_size = end_compute - begin_compute
        return new_request

    def _create_generation_request(self, request: LlmRequest,
                                   input_tokens: Any) -> LlmRequest:
        """Create a generation request when no tokens were accepted."""
        """当没有令牌被接受时创建 generation 请求。"""
        
        # 请求 draft 模型
        new_request = self._create_draft_request(request, input_tokens)
        # 设置状态为 GENERATION IN PROGRESS
        new_request.state = LlmRequestState.GENERATION_IN_PROGRESS
        return new_request

    def _create_accepted_tokens_request(self, request: LlmRequest,
                                        input_tokens: Any,
                                        num_accepted_tokens: int) -> LlmRequest:
        """
        Create a chunked context request for accepted tokens.
        Only applicable if the draft model needs to recompute KV cache for accepted tokens (e.g. eagle 3)
        
        为已接受的令牌创建分块上下文请求。
        仅当草案模型需要重新计算已接受令牌的KV缓存时适用（如EAGLE3）。
        """
        
        # 请求 draft 模型
        new_request = self._create_draft_request(request, input_tokens)
        
        new_request.context_chunk_size = num_accepted_tokens + 1   # 块大小包括新令牌
        new_request.context_current_position = len(input_tokens) - num_accepted_tokens - 1   # 当前位置
        return new_request

    def _create_draft_request_for_request(
            self, request: LlmRequest) -> Optional[LlmRequest]:
        """Create a draft request based on the original request state."""
        """根据原始请求状态创建草案请求。"""
        
        # 初始化请求的草案令牌跟踪
        num_draft_tokens, num_accepted_tokens = self._initialize_draft_tokens(
            request)
        
         # 获取草案模型的输入提示
        input_tokens = get_draft_model_prompt(self.spec_config.spec_dec_mode,
                                              request.get_tokens(0))

        # First time seeing this request - context request
        # 首次见到该请求 - 创建 context 请求
        if request.max_beam_num_tokens - 1 == request.py_prompt_len:
            # This is the first time the draft model is seeing this request.
            # Prepare a context request. We discard the first token and take
            # the newly decoded one - this is the convention for EAGLE 2 and 3.
            # 草案模型首次处理该请求，准备 context 请求
            assert num_draft_tokens == 0
            return self._create_context_request(request, input_tokens)

        # No tokens accepted - generation request
        # 没有令牌被接受 - 创建 generation 请求
        elif num_accepted_tokens == 0:
            return self._create_generation_request(request, input_tokens)

        # Tokens accepted - chunked context request
        # 有令牌被接受 - 创建分块上下文请求
        else:
            return self._create_accepted_tokens_request(request, input_tokens,
                                                        num_accepted_tokens)

    def _add_to_draft_batch(self, draft_batch: ScheduledRequests,
                            draft_request: LlmRequest,
                            original_request: LlmRequest) -> None:
        """Add the draft request to the appropriate batch list."""
        """将草案请求添加到适当的批次列表。"""
        # Copy additional properties
        # 复制额外属性
        draft_request.py_stop_words_list = original_request.py_stop_words_list

        # Add to appropriate batch based on request type
        # 根据请求类型添加到相应批次
        if draft_request.state == LlmRequestState.GENERATION_IN_PROGRESS:
            draft_batch.generation_requests.append(draft_request)
        else:
            draft_batch.context_requests.append(draft_request)

    @nvtx_range("_prepare_draft_batch")
    def _prepare_draft_batch(
            self, scheduled_requests: ScheduledRequests) -> ScheduledRequests:
        """
        为草案模型引擎准备批次。草案 token 仅针对 generation 请求产生。
        
        请求准备方式：
        1. 草案引擎首次见到请求时，是 context 请求。
        2. 如果上次目标模型解码步骤接受了草案令牌，则是分块 context 请求。
        3. 否则，是 generation 请求。

        参数:
            scheduled_requests: 要准备草案批次的调度请求

        返回:
            ScheduledRequests: 准备好的草案批次
        
        Prepares a batch for the draft model engine. Draft tokens are only produced
        for generation requests.

        The requests are prepared as follows:
        1. The first time the draft engine sees a request, it's a context request.
        2. Otherwise, if draft tokens were accepted on the last target model decoding
        step, it's a chunked context request (we process all the accepted tokens together).
        3. Otherwise, it's a generation request.

        Args:
            scheduled_requests: The scheduled requests to prepare draft batch for

        Returns:
            ScheduledRequests: The prepared draft batch
        """
        try:
            draft_batch = ScheduledRequests()

            # 处理 context 请求
            for request in scheduled_requests.context_requests:
                if request.is_first_context_chunk:
                    # 忽略仍需 target 模型处理的请求
                    # Ignore requests which still need to be processed by the target model.
                    continue

                # We hit this path if we're doing chunked prefill. The target model processed
                # a prefill chunk on the last iteration. Now, we need to fill in the KV cache
                # for the draft model too.
                # 处理 prefill chunk：target 模型处理了 prefill chunk，现在需要填充草案模型的KV缓存
                all_tokens = request.get_tokens(0)
                input_tokens = get_draft_model_prompt(
                    self.spec_config.spec_dec_mode, all_tokens)

                new_request = self._create_context_request(
                    request, input_tokens)
                self._add_to_draft_batch(draft_batch, new_request, request)
            # 处理generation请求
            for request in scheduled_requests.generation_requests:
                if request.py_draft_pages_allocated == 0:
                    # No space for draft tokens
                    # 没有空间存储草案令牌
                    continue
                # Stop drafting when we hit the max seqlen. We still need dummy draft
                # tokens attached to the requests to make sure everything works properly
                # with CUDA graph. These dummy tokens are already added by
                # _prepare_draft_requests to make the KV cache/scheduler aware of the fact
                # that we want to do spec decoding, so no need to do anything else here.
                # This makes the perf for this case suboptimal, but that's OK - this is
                # a corner case for weird models like the llama 3.1 8b EAGLE3 implementation.
                
                 # 达到最大序列长度时停止草案
                if request.max_beam_num_tokens - 1 >= self.draft_model_engine.max_seq_len:
                    continue

                draft_request = self._create_draft_request_for_request(request)
                if draft_request is not None:
                    self._add_to_draft_batch(draft_batch, draft_request,
                                             request)

            return draft_batch

        except Exception as e:
            logger.error(f"Error in _prepare_draft_batch: {str(e)}")
            traceback.print_exc()
            raise e

    def _should_disable_cuda_graph(
            self, previous_batch: Optional[SampleState]) -> bool:
        """Check if CUDA graph should be disabled for the current forward pass."""
        """检查当前前向传播是否应禁用CUDA图。"""
        if previous_batch is not None:
            return False
        # 需要重新计算KV缓存的模式禁用CUDA图
        return self.spec_config.spec_dec_mode.needs_kv_cache_recompute()

    def _forward_draft_model(
            self,
            draft_batch: ScheduledRequests,
            resource_manager: ResourceManager,
            previous_batch: Optional[SampleState] = None) -> Dict[str, Any]:
        """Forward pass through the draft model."""
        """草案模型的前向传播。"""
        
        if self._should_disable_cuda_graph(previous_batch):  # 关闭 cuda graph
            # 禁用CUDA图
            with self.draft_model_engine.no_cuda_graph():
                # draft 模型的 forward
                outputs = self.draft_model_engine.forward(
                    draft_batch, resource_manager)
        else:
            new_tensors_device = previous_batch.device if previous_batch else None
            outputs = self.draft_model_engine.forward(
                draft_batch,
                resource_manager,
                new_tensors_device=new_tensors_device)

        # Handle d2t data if available
        # 处理可选的d2t数据
        if hasattr(self.draft_model_engine.model.model, 'd2t'):
            outputs['d2t'] = self.draft_model_engine.model.model.d2t.data

        return outputs

    def _sample_async(self, draft_batch: ScheduledRequests,
                      outputs: Dict[str, Any]) -> Optional[SampleState]:
        """Sample tokens from draft model outputs."""
        """从草案模型输出中采样 tokens。"""
        try:
            if self.sampler is not None:
                # 计算 contexts requests 的 logits 大小的前缀和
                num_context_logits_prefix_sum = [0]
                prefix_sum = 0
                for request in draft_batch.context_requests:
                    prefix_sum += request.context_chunk_size if request.py_return_context_logits else 1
                    num_context_logits_prefix_sum.append(prefix_sum)
                # 处理logits
                HandleLogits()(
                    draft_batch.context_requests,
                    draft_batch.generation_requests, 
                    outputs["logits"],
                    self.sampler.beam_width(draft_batch.all_requests()),
                    num_context_logits_prefix_sum,
                    self.sampler.is_generation_model())
                # 异步采样
                return self.sampler.sample_async(draft_batch, outputs,
                                                 num_context_logits_prefix_sum)
            return None
        except Exception as e:
            logger.error(f"Error in sampling: {str(e)}")
            return None

    def _update_request_states(self,
                               scheduled_requests: ScheduledRequests) -> None:
        """Update request states after processing."""
        """处理完成后更新请求状态。"""
        for request in scheduled_requests.context_requests:
            if request.state != LlmRequestState.GENERATION_COMPLETE:
                request.move_to_next_context_chunk()
            if request.context_remaining_length == 0:
                request.state = LlmRequestState.GENERATION_IN_PROGRESS

    def _update_requests(self, sample_state: SampleState) -> None:
        """Update requests with sample state."""
        if self.sampler is not None:
            self.sampler.update_requests(sample_state)

    def _process_decoded_tokens(
            self, draft_batch: ScheduledRequests,
            req_id_to_old_request: Dict[int, LlmRequest]) -> List[LlmRequest]:
        """Process decoded tokens and determine which requests to continue processing."""
        new_requests = []
        for req in draft_batch.all_requests():
            target_model_req = req_id_to_old_request[req.py_request_id]
            if target_model_req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                # This is a chunked prefill request and we have more prefill chunks
                # to process. Defer adding draft tokens until the whole prompt is processed.
                self.draft_seq_slot_manager.free_resources(req)
                continue

            target_model_req.py_draft_tokens.append(req.get_last_tokens(0))
            if self._request_draft_logits:
                target_model_req.py_draft_logits = req.py_result.generation_logits
            if req.state != LlmRequestState.GENERATION_COMPLETE and len(
                    target_model_req.py_draft_tokens
            ) < target_model_req.py_draft_pages_allocated:
                new_requests.append(req)
            else:
                self.draft_seq_slot_manager.free_resources(req)

        return new_requests

    def _execute_guided_decoder(self,
                                scheduled_batch: ScheduledRequests,
                                logits: torch.Tensor,
                                d2t: Optional[torch.Tensor] = None):
        if self.guided_decoder is not None:
            self.guided_decoder.build(scheduled_batch)
            self.guided_decoder.execute(scheduled_batch, logits, d2t=d2t)

    @nvtx_range("prepare_draft_tokens")
    def prepare_draft_tokens(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        """
        Prepare draft tokens for the scheduled requests.

        Args:
            scheduled_requests: The scheduled requests for this iteration
            resource_manager: The resource manager for this iteration
        """
        if not self.draft_model_engine:
            raise ValueError("Draft model engine is not set")

        if resource_manager is None:
            raise ValueError("Resource manager is required")

        try:
            draft_batch = self._prepare_draft_batch(scheduled_requests)

            if draft_batch.batch_size == 0:
                return

            self.draft_seq_slot_manager.prepare_resources(draft_batch)

            req_id_to_old_request = {
                req.py_request_id: req
                for req in scheduled_requests.all_requests()
            }

            # Initial forward pass
            outputs = self._forward_draft_model(draft_batch, resource_manager)
            self._execute_guided_decoder(draft_batch,
                                         outputs['logits'],
                                         d2t=outputs.get('d2t'))
            sample_state = self._sample_async(draft_batch, outputs)
            previous_batch = sample_state

            self._update_request_states(draft_batch)

            # Convert context requests to generation requests
            draft_batch.generation_requests = draft_batch.context_requests + draft_batch.generation_requests
            draft_batch.context_requests = []

            # Generate remaining draft tokens iteratively
            for i in range(self.max_draft_tokens - 1):
                if len(draft_batch.generation_requests) == 0:
                    break

                outputs = self._forward_draft_model(draft_batch,
                                                    resource_manager,
                                                    previous_batch)
                if previous_batch is not None:
                    self._update_requests(previous_batch)
                self._execute_guided_decoder(draft_batch,
                                             outputs['logits'],
                                             d2t=outputs.get('d2t'))
                sample_state = self._sample_async(draft_batch, outputs)
                self._update_request_states(draft_batch)
                if previous_batch is not None:
                    new_requests = self._process_decoded_tokens(
                        previous_batch.scheduled_requests,
                        req_id_to_old_request)
                else:
                    new_requests = []
                draft_batch.generation_requests = new_requests
                previous_batch = sample_state

            # Final cleanup
            if previous_batch is not None:
                self._update_requests(previous_batch)
                self._process_decoded_tokens(previous_batch.scheduled_requests,
                                             req_id_to_old_request)

            if self.guided_decoder is not None:
                self.guided_decoder.rollback_draft_tokens(scheduled_requests)

        except Exception as e:
            traceback.print_exc()
            error_msg = str(e)
            logger.error(f"Encountered an error in decode: {error_msg}")
            raise e
