# batata-ai

Rust AI 底座平台 — 本地推理、多 Provider、MCP 协议集成。

基于 [candle](https://github.com/huggingface/candle) 实现本地模型推理，通过 [rig-core](https://github.com/0xPlaygrounds/rig) 对接远程 AI 服务，使用 [rmcp](https://github.com/modelcontextprotocol/rust-sdk) 提供 MCP 协议支持。

## 支持的本地模型

所有模型均使用 GGUF Q4 量化格式，首次运行时自动从 HuggingFace Hub 下载。

| 模型 | 别名 | 参数量 | 下载大小 | 最低内存 | 适用场景 |
|------|------|--------|----------|----------|----------|
| [Qwen2-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF) | `qwen2` | 1.5B | ~1.1 GB | 2 GB | 轻量对话、资源受限环境 |
| [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF) | `qwen3` | 1.7B | ~1.2 GB | 2 GB | 轻量对话、思考模式 |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B-GGUF) | `qwen3-4b` | 4B | ~2.5 GB | 4 GB | 通用对话、代码生成 |
| [Gemma-3-1B-IT](https://huggingface.co/google/gemma-3-1b-it-qat-q4_0-gguf) | `gemma3` | 1B | ~0.7 GB | 1.5 GB | 超轻量、边缘设备 |
| [Gemma-3-4B-IT](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) | `gemma3-4b` | 4B | ~2.5 GB | 4 GB | 通用对话、Google 优化 |
| [Phi-3-mini-4K-Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) | `phi3` | 3.8B | ~2.3 GB | 4 GB | 通用对话、代码生成 |
| [Llama-3-8B-Instruct](https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF) | `llama3` | 8B | ~4.7 GB | 8 GB | 通用对话、推理 |

## 多模态能力

除了文本对话，batata-ai 还支持语音识别和图像理解。

### 语音识别 (Whisper)

基于 OpenAI Whisper 模型，支持多语言语音转文字。

| 模型 | 大小 | 内存需求 | 速度 |
|------|------|----------|------|
| Whisper Tiny | ~75 MB | 1 GB | 最快 |
| Whisper Base | ~140 MB | 1 GB | 快 |
| Whisper Small | ~460 MB | 2 GB | 中等 |
| Whisper Medium | ~1.5 GB | 4 GB | 较慢 |
| Whisper Large v3 Turbo | ~3 GB | 6 GB | 慢但最准 |

```rust
use batata_ai::local::whisper::{WhisperModel, WhisperSize};
use batata_ai::local::model::resolve_device;

let device = resolve_device(false)?; // CPU
let mut whisper = WhisperModel::download_and_load(WhisperSize::Base, &device)?;

// 从 WAV 文件转写
let result = whisper.transcribe_file(Path::new("audio.wav"))?;
println!("Transcription: {}", result.text);

// 从 PCM 数据转写
use batata_ai::multimodal::AudioData;
let result = whisper.transcribe(AudioData {
    samples: pcm_f32_samples,
    sample_rate: 16000,
})?;
```

### 图像描述 (BLIP)

基于 Salesforce BLIP 模型，自动为图片生成描述文字。

```rust
use batata_ai::local::blip::BlipModel;
use batata_ai::multimodal::ImageData;

let device = resolve_device(false)?;
let mut blip = BlipModel::download_and_load(&device)?;

let caption = blip.caption(ImageData {
    bytes: rgb_bytes,   // RGB pixel data
    width: 640,
    height: 480,
})?;
println!("Caption: {}", caption);
```

BLIP image captioning large 模型大小约 ~1.8 GB。

### 文本/图像嵌入 (CLIP)

基于 OpenAI CLIP ViT-B/32，生成 512 维语义向量，可用于 RAG、语义搜索、图文匹配。

```bash
cargo run --example clip_embed
```

```rust
use batata_ai::local::clip::ClipModel;

let device = resolve_device(false)?;
let clip = ClipModel::download_and_load(&device)?;

// 文本嵌入
let embeddings = clip.embed_texts(&vec![
    "a photo of a cat".into(),
    "Rust programming".into(),
])?;

// 语义相似度
let sim = ClipModel::cosine_similarity(&embeddings[0], &embeddings[1]);

// 图像嵌入
let img_embed = clip.embed_image(&image_data)?;
```

CLIP ViT-B/32 模型大小约 ~600 MB，嵌入维度 512。

### 多模态 Trait 体系

`batata-ai-core` 定义了统一的多模态 trait，便于扩展：

| Trait | 用途 | 已实现 |
|-------|------|--------|
| `SpeechToText` | 语音转文字 | Whisper |
| `TextToSpeech` | 文字转语音 | - |
| `TextToImage` | 文字生成图像 | - |
| `ImageToText` | 图像描述/VQA | BLIP |
| `ObjectDetection` | 目标检测 | - |
| `Embedding` | 向量嵌入 | CLIP ViT-B/32 |

candle 已支持但尚未集成的模型：Stable Diffusion（图像生成）、Segment Anything（图像分割）、Parler TTS（语音合成）、YOLO（目标检测）。

### GPU 加速

通过 feature flag 启用：

```bash
# NVIDIA GPU (CUDA)
cargo run --features batata-ai-local/cuda --example local_chat

# Apple Silicon (Metal)
cargo run --features batata-ai-local/metal --example local_chat

# Intel/AMD CPU 加速 (macOS Accelerate)
cargo run --features batata-ai-local/accelerate --example local_chat
```

默认使用 CPU 推理，无需额外配置。

## 快速开始

### 环境要求

- Rust 1.85+ (edition 2024)
- 至少 4 GB 可用内存（Phi-3-mini）或 2 GB（Qwen2）
- 网络连接（首次下载模型）

### 本地对话

```bash
cargo run --example local_chat
```

```rust
use batata_ai::core::message::{ChatRequest, Message};
use batata_ai::core::provider::Provider;
use batata_ai::local::provider::LocalProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 选择模型: "phi3", "llama3", "qwen2", "qwen3"
    let provider = LocalProvider::new("phi3");

    let request = ChatRequest {
        messages: vec![
            Message::system("You are a helpful assistant."),
            Message::user("What is Rust?"),
        ],
        model: None,
        temperature: Some(0.7),
        max_tokens: Some(128),
    };

    let response = provider.chat(request).await?;
    println!("{}", response.content);
    Ok(())
}
```

### 流式输出

```bash
cargo run --example local_stream
```

```rust
use std::io::Write;
use futures::StreamExt;
use batata_ai::core::message::{ChatRequest, Message};
use batata_ai::core::provider::Provider;
use batata_ai::local::provider::LocalProvider;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let provider = LocalProvider::new("phi3");

    let request = ChatRequest {
        messages: vec![
            Message::system("You are a helpful assistant."),
            Message::user("Explain Rust ownership in 3 sentences."),
        ],
        model: None,
        temperature: Some(0.7),
        max_tokens: Some(200),
    };

    let mut stream = provider.stream_chat(request).await?;

    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                print!("{token}");
                std::io::stdout().flush()?;
            }
            Err(e) => eprintln!("\nError: {e}"),
        }
    }
    Ok(())
}
```

### MCP Server (stdio)

适用于 Claude Desktop 等 CLI 工具：

```bash
cargo run --example mcp_server
```

在 Claude Desktop 中配置：

```json
{
  "mcpServers": {
    "batata-ai": {
      "command": "cargo",
      "args": ["run", "--release", "--example", "mcp_server"],
      "cwd": "/path/to/batata-ai"
    }
  }
}
```

### MCP Server (HTTP/SSE)

适用于 Web 客户端和远程调用：

```bash
cargo run --example mcp_http_server

# 自定义端口
BIND_ADDR=0.0.0.0:3000 cargo run --example mcp_http_server
```

MCP 端点地址：`http://127.0.0.1:8080/mcp`

测试连接：

```bash
# 初始化会话
curl -X POST http://127.0.0.1:8080/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-03-26",
      "capabilities": {},
      "clientInfo": {"name": "test", "version": "1.0"}
    }
  }'

# 列出工具（使用返回的 Mcp-Session-Id）
curl -X POST http://127.0.0.1:8080/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H 'Mcp-Session-Id: <session-id>' \
  -d '{"jsonrpc": "2.0", "id": 2, "method": "tools/list"}'
```

也可以在代码中嵌入到现有 axum 应用：

```rust
let server = BatataMcpServer::new(providers, skills);
let mcp_router = server.into_router();

let app = axum::Router::new()
    .route("/health", axum::routing::get(|| async { "ok" }))
    .merge(mcp_router);
```

### MCP Tools

| Tool | 描述 | 模型 |
|------|------|------|
| `chat` | 向 AI Provider 发送消息并获取回复 | LLM |
| `transcribe` | 语音转文字（WAV 文件） | Whisper Base |
| `caption` | 图像描述（PPM 文件） | BLIP Large |
| `embed_text` | 文本语义嵌入（512 维向量） | CLIP ViT-B/32 |
| `list_providers` | 列出所有注册的 Provider | - |
| `list_skills` | 列出所有注册的 Skill | - |
| `execute_skill` | 按名称执行指定的 Skill | - |

多模态模型在首次调用时自动下载，后续调用直接使用缓存。

### OpenRouter (免费模型)

[OpenRouter](https://openrouter.ai/) 提供大量免费开源模型，注册即可获取 API Key。

```rust
use batata_ai::provider::openrouter::{OpenRouterProvider, self};

let provider = OpenRouterProvider::new("sk-or-xxx")?
    .with_model(openrouter::QWEN3_6_PLUS_FREE);
```

可用的免费模型常量：

| 常量 | 模型 | 参数量 | 上下文 |
|------|------|--------|--------|
| `QWEN3_6_PLUS_FREE` | Qwen3.6 Plus | - | 1M |
| `QWEN3_CODER_FREE` | Qwen3 Coder | 480B MoE | 262K |
| `QWEN3_NEXT_80B_FREE` | Qwen3 Next 80B | 80B MoE | 262K |
| `LLAMA_3_3_70B_FREE` | Llama 3.3 70B | 70B | 65K |
| `HERMES_3_405B_FREE` | Hermes 3 405B | 405B | 131K |
| `GPT_OSS_120B_FREE` | GPT OSS 120B | 120B | 131K |
| `GPT_OSS_20B_FREE` | GPT OSS 20B | 20B | 131K |
| `NEMOTRON_SUPER_120B_FREE` | Nemotron 3 Super | 120B MoE | 262K |
| `NEMOTRON_NANO_9B_FREE` | Nemotron Nano 9B | 9B | 128K |
| `GEMMA_3_27B_FREE` | Gemma 3 27B | 27B | 131K |
| `GEMMA_3_12B_FREE` | Gemma 3 12B | 12B | 32K |
| `GEMMA_3_4B_FREE` | Gemma 3 4B | 4B | 32K |
| `GEMMA_3N_E4B_FREE` | Gemma 3N E4B | 4B | 8K |
| `GEMMA_3N_E2B_FREE` | Gemma 3N E2B | 2B | 8K |
| `DOLPHIN_MISTRAL_24B_FREE` | Dolphin Mistral 24B | 24B | 32K |
| `LLAMA_3_2_3B_FREE` | Llama 3.2 3B | 3B | 131K |
| `LFM_1_2B_FREE` | LFM 2.5 1.2B | 1.2B | 32K |
| `MINIMAX_M2_5_FREE` | MiniMax M2.5 | - | 196K |
| `STEP_3_5_FLASH_FREE` | Step 3.5 Flash | - | 256K |
| `GLM_4_5_AIR_FREE` | GLM 4.5 Air | - | 131K |
| `OPENROUTER_FREE` | Auto (随机路由) | - | varies |

也可以传入任意 OpenRouter 模型 ID：

```rust
let provider = OpenRouterProvider::new("sk-or-xxx")?
    .with_model("meta-llama/llama-3.3-70b-instruct:free");
```

### 其他远程 Provider (OpenAI / Anthropic / Ollama)

```rust
use batata_ai::provider::openai::OpenAiProvider;
use batata_ai::provider::ollama::OllamaProvider;

// OpenAI
let openai = OpenAiProvider::new("sk-xxx")?;

// Ollama (本地运行)
let ollama = OllamaProvider::new().with_model("llama3");

// 注册到 ProviderRegistry
let mut registry = ProviderRegistry::new();
registry.register(Box::new(openai));
registry.register(Box::new(ollama));
registry.set_default("openai");
```

## 项目结构

```
batata-ai/
├── Cargo.toml                    # Workspace 根
├── src/lib.rs                    # 统一 re-export
├── crates/
│   ├── batata-ai-core/           # 核心 trait 和类型
│   │   ├── provider.rs           #   Provider trait + ProviderRegistry
│   │   ├── skill.rs              #   Skill trait + SkillRegistry
│   │   ├── message.rs            #   ChatRequest/ChatResponse/Message
│   │   ├── multimodal.rs         #   多模态 trait (STT/TTS/Image/Embedding)
│   │   ├── config.rs             #   BatataConfig
│   │   └── error.rs              #   BatataError
│   │
│   ├── batata-ai-local/          # 本地推理引擎 (candle)
│   │   ├── models.rs             #   多模型支持 (Phi3/Llama/Qwen/Gemma)
│   │   ├── provider.rs           #   LocalProvider (实现 Provider trait)
│   │   ├── whisper.rs            #   Whisper 语音识别
│   │   ├── blip.rs               #   BLIP 图像描述
│   │   ├── phi3.rs               #   Phi-3 专用实现
│   │   ├── model.rs              #   设备管理、HF Hub 下载
│   │   └── inference.rs          #   生成参数
│   │
│   ├── batata-ai-provider/       # 远程 Provider (rig-core)
│   │   ├── openai.rs             #   OpenAI / 兼容 API
│   │   ├── anthropic.rs          #   Claude
│   │   ├── ollama.rs             #   Ollama
│   │   └── openrouter.rs         #   OpenRouter (21 个免费模型)
│   │
│   ├── batata-ai-mcp/            # MCP 协议 (rmcp)
│   │   ├── server.rs             #   BatataMcpServer
│   │   └── tools.rs              #   MCP Tool 参数类型
│   │
│   └── batata-ai-prompt/         # Prompt 模板引擎
│       ├── template.rs           #   PromptTemplate (变量替换)
│       └── store.rs              #   PromptStore (持久化)
│
└── examples/
    ├── local_chat.rs             # 本地对话示例
    ├── local_stream.rs           # 流式输出示例
    ├── mcp_server.rs             # MCP Server 示例 (stdio)
    ├── mcp_http_server.rs        # MCP Server 示例 (HTTP/SSE)
    ├── whisper_transcribe.rs     # Whisper 语音识别示例
    ├── clip_embed.rs             # CLIP 文本嵌入示例
    └── download_model.rs         # 模型预下载工具
```

## 注意事项

### 模型下载

- 首次运行会自动从 HuggingFace Hub 下载模型，需要网络连接
- 模型缓存在 `~/.cache/huggingface/` 目录，后续运行无需重新下载
- 如果 HuggingFace 无法直接访问，可设置镜像：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  ```

#### 预下载模型到本地目录

```bash
# 下载到 ./models/ 目录
cargo run --example download_model -- phi3

# 下载到指定目录
cargo run --example download_model -- qwen2 /opt/batata-ai/models

# 查看所有可下载的模型
cargo run --example download_model
```

#### 从本地文件加载（离线使用）

```rust
// 从预下载的本地文件加载，不需要网络
let provider = LocalProvider::from_local(
    "phi3",                              // 模型架构
    "./models/Phi-3-mini-4k-instruct-q4.gguf",  // GGUF 模型文件
    "./models/tokenizer.json",           // 分词器文件
);
```

适用于：
- 离线 / 内网部署
- Docker 镜像（构建时下载，运行时加载）
- CI/CD 环境

### 内存使用

- Q4 量化模型的内存使用约为下载大小的 1.2-1.5 倍
- Phi-3-mini (3.8B Q4): 运行时约 3-4 GB 内存
- Qwen2-1.5B (Q4): 运行时约 1.5-2 GB 内存
- Llama-3-8B (Q4): 运行时约 6-8 GB 内存
- 推理过程中 KV Cache 会随对话长度增长，长对话需要更多内存

### 性能

- CPU 推理速度：Phi-3-mini 约 5-15 tokens/s（取决于 CPU）
- Apple Silicon (Metal): 约 30-60 tokens/s
- NVIDIA GPU (CUDA): 约 50-100+ tokens/s
- 首次推理（prompt processing）较慢，后续 token 生成较快
- 建议生产环境使用 `--release` 编译：
  ```bash
  cargo run --release --example local_chat
  ```

### Qwen3 兼容性

- Qwen3 使用 Qwen2 架构加载器（架构基本兼容）
- 如果遇到加载失败，可能是 GGUF metadata key 不兼容，需等待 candle 官方支持
- 建议优先使用 Qwen2 作为稳定选项

### Llama-3 访问权限

- Llama-3 模型需要先在 HuggingFace 上接受 Meta 的许可协议
- 访问 [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 接受协议
- 然后设置 HF Token：
  ```bash
  export HF_TOKEN=hf_xxxxx
  ```

### MCP Server

- MCP Server 使用 stdio 传输，日志输出到 stderr
- 调试时可使用环境变量：
  ```bash
  RUST_LOG=info cargo run --example mcp_server
  ```

## 技术栈

| 组件 | 依赖 | 用途 |
|------|------|------|
| 本地推理 | candle-core, candle-transformers | GGUF 模型加载与推理 |
| 远程 Provider | rig-core | OpenAI/Anthropic/Ollama API |
| MCP 协议 | rmcp | MCP Server/Client |
| 异步运行时 | tokio | 异步 I/O |
| 序列化 | serde, serde_json | JSON 处理 |
| 分词器 | tokenizers | HF Tokenizer |
| 模型下载 | hf-hub | HuggingFace Hub API |

## License

MIT OR Apache-2.0
