import { defineConfig } from 'vitepress'

const base = process.env.VITEPRESS_BASE || '/'

export default defineConfig({
  lang: 'zh-CN',
  title: '从零实现 LLM 推理引擎',
  description: '20步学懂 vLLM 核心原理，从 Tokenizer 到 PagedAttention 到 HTTP 服务',
  base,

  sitemap: {
    hostname: 'https://mini-vllm-tutorial.vercel.app',
  },

  head: [
    ['meta', { name: 'author', content: 'happydog-intj' }],
    ['meta', { name: 'keywords', content: 'vLLM,LLM推理,KV Cache,PagedAttention,FlashAttention,CUDA Graph,Tensor Parallel,推理引擎,大模型推理' }],
    ['meta', { property: 'og:type', content: 'website' }],
    ['meta', { property: 'og:locale', content: 'zh_CN' }],
    ['meta', { property: 'og:site_name', content: '从零实现 LLM 推理引擎' }],
    ['meta', { property: 'og:title', content: '从零实现 LLM 推理引擎：20步学懂 vLLM 核心原理' }],
    ['meta', { property: 'og:description', content: '从 Tokenizer 到 PagedAttention 到 HTTP 服务，每步可运行的代码 + ASCII 图解' }],
  ],

  themeConfig: {
    nav: [
      { text: '教程', link: '/step01_tokenizer/' },
      { text: '进阶', link: '/advanced/' },
      { text: 'GitHub', link: 'https://github.com/happydog-intj/mini-vllm-tutorial' },
    ],

    sidebar: sidebar(),

    outline: {
      level: [2, 3],
      label: '目录',
    },

    search: {
      provider: 'local',
    },

    docFooter: {
      prev: '上一章',
      next: '下一章',
    },

    lastUpdated: {
      text: '最后更新',
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/happydog-intj/mini-vllm-tutorial' },
    ],
  },
})

function sidebar() {
  return [
    {
      text: 'Phase 0 · 基础概念',
      items: [
        { text: '01. Tokenizer：文字变数字', link: '/step01_tokenizer/' },
        { text: '02. Embedding：数字变向量', link: '/step02_embedding/' },
        { text: '03. Attention：注意力机制', link: '/step03_attention/' },
        { text: '04. Transformer：搭建完整模型', link: '/step04_transformer/' },
      ],
    },
    {
      text: 'Phase 1 · 朴素推理与采样',
      items: [
        { text: '05. 朴素推理：逐 token 生成', link: '/step05_naive/' },
        { text: '06. 采样策略', link: '/step06_sampler/' },
      ],
    },
    {
      text: 'Phase 2 · KV Cache',
      items: [
        { text: '07. 单请求 KV Cache', link: '/step07_kvcache_for_single_request/' },
        { text: '08. 多请求 Static Batching', link: '/step08_kvcache_static_batching_for_multi_requests/' },
      ],
    },
    {
      text: 'Phase 3 · 调度',
      items: [
        { text: '09. Continuous Batching', link: '/step09_kvcache_continuous_batching_for_multi_requests/' },
        { text: '10. Chunked Prefill', link: '/step10_chunked_prefill/' },
        { text: '11. Preemption：抢占避免 OOM', link: '/step11_preemption/' },
      ],
    },
    {
      text: 'Phase 4 · PagedAttention',
      items: [
        { text: '12. PagedAttention：分页内存', link: '/step12_paged_attention/' },
        { text: '13. Prefix Caching', link: '/step13_prefix_cache/' },
      ],
    },
    {
      text: 'Phase 5 · Paged Prefix Cache',
      collapsed: true,
      items: [
        { text: '14. Paged Prefix Cache', link: '/step14_paged_prefix_cache/' },
        { text: '14.1 向量化 KV 写入', link: '/step14_1_vectorized_kv_write/' },
        { text: '14.2 index_select gather', link: '/step14_2_index_select_gather/' },
        { text: '14.3 批量 Attention', link: '/step14_3_batched_attention/' },
        { text: '14.4 Causal Mask 优化', link: '/step14_4_causal_mask_opt/' },
        { text: '14.5 增量 Hash', link: '/step14_5_incremental_hash/' },
        { text: '14.6 BlockManager.release()', link: '/step14_6_block_manager_release/' },
        { text: '14.7 Batched Forward', link: '/step14_7_batched_forward/' },
      ],
    },
    {
      text: 'Phase 6 · 高性能内核',
      items: [
        { text: '15. FlashAttention 封装', link: '/step15_flash_attention/' },
        { text: '16. CUDA Graph', link: '/step16_cuda_graph/' },
      ],
    },
    {
      text: 'Phase 7 · 分布式',
      items: [
        { text: '17. Tensor Parallelism', link: '/step17_tensor_parallel/' },
      ],
    },
    {
      text: 'Phase 8 · 工程落地',
      items: [
        { text: '18. Benchmark：推理性能评测', link: '/step18_benchmark/' },
        { text: '19. Real Model：Qwen3', link: '/step19_real_model/' },
        { text: '20. HTTP Serve：OpenAI 兼容', link: '/step20_serve/' },
      ],
    },
    {
      text: '进阶系列',
      collapsed: true,
      items: [
        { text: '进阶系列简介', link: '/advanced/' },
        { text: 'adv01 量化 Quantization', link: '/advanced/adv01_quantization/' },
        { text: 'adv02 采样进阶', link: '/advanced/adv02_sampling_advanced/' },
        { text: 'adv03 投机解码', link: '/advanced/adv03_speculative_decoding/' },
        { text: 'adv04 Flash-Decoding', link: '/advanced/adv04_flash_decoding/' },
        { text: 'adv05 Radix Prefix Cache', link: '/advanced/adv05_radix_prefix_cache/' },
        { text: 'adv06 Pipeline Parallel', link: '/advanced/adv06_pipeline_parallel/' },
        { text: 'adv07 Sequence Parallel', link: '/advanced/adv07_sequence_parallel/' },
        { text: 'adv08 Data Parallel + DPLB', link: '/advanced/adv08_data_parallel_dplb/' },
        { text: 'adv09 TBO/DBO 重叠', link: '/advanced/adv09_tbo_dbo_overlap/' },
        { text: 'adv10 PD Disaggregation', link: '/advanced/adv10_pd_disaggregation/' },
        { text: 'adv11 AFD 注意力前馈分离', link: '/advanced/adv11_afd_attention_ffn/' },
        { text: 'adv12 MoE + EPLB', link: '/advanced/adv12_moe_eplb/' },
        { text: 'adv13 Linear Attention', link: '/advanced/adv13_linear_attention/' },
        { text: 'adv14 Multi-LoRA', link: '/advanced/adv14_multi_lora/' },
        { text: 'adv15 Guided Decoder', link: '/advanced/adv15_guided_decoder/' },
        { text: 'adv16 Function Call', link: '/advanced/adv16_function_call/' },
        { text: 'adv17 Logits Tricks', link: '/advanced/adv17_logits_tricks/' },
      ],
    },
    {
      text: '附录',
      items: [
        { text: '目录：为什么需要推理引擎', link: '/CONTENTS' },
        { text: '推理优化总结', link: '/summary_overview' },
      ],
    },
  ]
}
