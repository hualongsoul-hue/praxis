# Praxis 公共符号、多模态输入与测试严谨性设计

## 背景

Praxis 当前公开会话入口只接受 `str`。消息模型虽然定义了图片内容块，但图片无法从
`AgentSession.run()` 进入编排链路；文件、音频和视频没有输入模型。项目自有 Python
代码还存在单下划线方法、属性和模块变量，测试中也有直接访问这些实现细节的情况。
现有 `tests/e2e` 中相当一部分测试替换了模型网关，只能证明组件协作，不能证明 SDK、
序列化、HTTP 适配器和模型响应之间的真实契约。

本设计不保留旧 API 兼容层，目标是在 Windows 和 Linux 上提供严格类型化、安全且可
验证的多模态输入，并建立能够持续阻止私有符号和过拟合测试回归的工程门禁。

## 设计目标

- 项目自有 Python 源码、测试、脚本和示例不定义或访问单下划线私有变量、方法或类。
- Python 规定的双下划线协议方法（如 `__init__`、`__aenter__`）不视为私有 API。
- `AgentSession.run()` 和 `run_stream()` 同时接受字符串与类型化 `UserInput`。
- 输入支持文本、图片、音频、视频和一般文件，来源支持内存字节、本地路径和远程 URL。
- 模型能力、输入安全策略和供应商消息序列化相互独立；不支持的模态在网络请求前失败。
- 二进制内容只在当前运行期间存在，不进入日志、审计、检查点、记忆或后续持久化历史。
- 流式与非流式路径使用同一输入规范化和清理流程。
- 使用行为、属性、变形、HTTP 契约和真实端点测试减少对实现细节的依赖。

## 非目标

- 不实现 OCR、语音转写、视频抽帧或任意文件内容解析框架。
- 不假定指定模型端点支持所有模态；端点能力必须由真实请求确认后显式配置。
- 不提供旧的 `user_message` 字段、`supports_vision()` 或单下划线名称兼容别名。
- 不把二进制输入保存到检查点后再恢复。

## 公共符号规则

仓库增加 AST 门禁，扫描 `src/`、`tests/`、`scripts/` 和 `examples/` 中的 Python 文件。
以下项目自有符号一律禁止：

- 名称以单个 `_` 开头的函数、异步函数和类；
- 模块级赋值、注解赋值和导入别名中的单下划线名称；
- `self._name`、`cls._name` 以及项目对象的 `object._name` 属性访问；
- 测试为了断言实现细节而访问的私有状态。

双下划线协议名称允许保留。第三方库只有公开 API 可以被调用；若第三方确实没有公共
能力，则通过项目公共适配器隔离，而不是把第三方私有名称扩散到业务代码。所有现有
私有状态直接改为公共、含义明确的名称，不保留转发属性。

## 公共输入模型

新增以下稳定类型：

```python
InputValue = str | UserInput

class UserInput(BaseModel):
    text: str = ""
    parts: tuple[ImageInput | AudioInput | VideoInput | FileInput, ...] = ()

class ImageInput(AttachmentInput): ...
class AudioInput(AttachmentInput): ...
class VideoInput(AttachmentInput): ...
class FileInput(AttachmentInput): ...
```

每种附件提供 `from_bytes()`、`from_path()` 和 `from_url()` 公共构造器。来源类型被显式
记录，普通字符串不会被猜测为路径或 URL。`UserInput` 必须至少包含非空文本或一个附件。

公开调用保持字符串便捷路径：

```python
await session.run("普通文本")

await session.run(UserInput(
    text="分析这些资料",
    parts=(
        ImageInput.from_path("diagram.png"),
        AudioInput.from_bytes(audio, media_type="audio/wav"),
        VideoInput.from_url(video_url),
        FileInput.from_path("report.pdf"),
    ),
))
```

## 安全解析与规范化

`InputConfig` 是不可变严格配置，包含授权根目录、远程 URL 开关、私网开关、逐附件与
总字节限制、重定向限制、下载超时和各模态 MIME 白名单。默认值：

- `allowed_paths=[]`，因此本地路径默认拒绝；
- `remote_enabled=False`，因此远程附件默认拒绝；
- `allow_private_networks=False`；
- `max_attachment_bytes=20_000_000`；
- `max_total_bytes=50_000_000`；
- `max_redirects=5`；
- `remote_timeout=30.0` 秒。

`InputResolver.resolve()` 是唯一读取二进制来源的组件：

1. 本地路径先解析为绝对路径，再验证位于授权根目录内；目录、符号链接越界和不存在
   的文件返回类型化异常。
2. 内存字节必须显式提供 MIME；路径按扩展名推断 MIME；远程响应使用去参数后的
   `Content-Type`。
3. MIME 必须属于输入种类的白名单；文件名只保留安全 basename。
4. URL 只允许 HTTP/HTTPS、禁止 URL 凭据，并在每次重定向前重新解析 DNS；默认拒绝
   回环、私网、链路本地、保留和非全局地址。
5. 所有来源以流式/分块方式计数，在超过单附件或总限制时立即停止。
6. 解析结果转换为 OpenAI 兼容内容块；图片、视频和文件使用 data URL，音频使用
   `input_audio` 的 Base64 数据与格式字段。

解析结果 `ResolvedUserInput` 同时包含供应商内容和安全文本投影。文本投影只含用户文本、
附件类型、文件名、MIME 和字节数，不含 URL 查询、原始字节或 Base64。

## 能力模型与运行链路

`ModelDeployment` 使用嵌套的 `ModelCapabilities`：

```python
class ModelCapabilities(StrictConfigModel):
    image: bool = False
    audio: bool = False
    video: bool = False
    file: bool = False
```

`ModelGateway.capabilities(model_name=None)` 返回所选部署的不可变能力快照。视觉验证也使用
该快照，不再依赖单独的 `supports_vision()`。只要输入包含未声明支持的模态，解析器就在
调用模型前抛出 `UnsupportedInputModalityError`。

`Session` 在进入编排循环前异步解析输入。`TurnContext` 分离 `user_content` 与
`user_text`：前者供模型使用，后者供护栏、记忆检索、技能激活、计划生成和审计元数据
使用。结构化用户消息在同一次 Agent 工具循环的每次模型调用中保持可见；无论正常返回、
异常还是取消，`finally` 都会把历史中的结构化内容替换为安全文本投影。检查点只能看到
清理后的历史。

## 供应商内容格式

消息模型增加以下内容块：

- 图片：`{"type":"image_url","image_url":{"url":"data:..."}}`
- 音频：`{"type":"input_audio","input_audio":{"data":"...","format":"wav|mp3"}}`
- 视频：`{"type":"video_url","video_url":{"url":"data:..."}}`
- 文件：`{"type":"file","file":{"filename":"...","file_data":"data:..."}}`

这些格式属于适配器契约，不等价于端点能力。真实端点测试会分别发送最小有效样本；成功
的模态才能在示例部署中设为 `true`。端点拒绝的模态保留 SDK 支持，但示例配置保持
`false`，调用时返回本地、可诊断的能力异常。

## 测试策略

测试按以下层次重新组织：

1. 单元测试验证输入模型、路径/MIME/大小、能力门禁、清理和异常映射。
2. Hypothesis 属性测试生成边界大小、文件名、MIME 和内容组合；验证任意输入都不会让
   Base64 进入文本投影，字符串输入与 `UserInput(text=...)` 语义等价。
3. 变形测试验证流式与非流式路径向网关发送相同结构化消息，并在成功、异常、取消后
   得到相同的安全历史。
4. 标准库本地 OpenAI 兼容 HTTP Server 验证真实 JSON、SSE、工具调用和所有内容块经过
   LiteLLM 与 Runtime 的完整链路；测试结束必须关闭并 join 后台线程。
5. 原 `tests/e2e` 中替换网关的测试移动为 `tests/scenarios`；只有经过真实 HTTP 边界的
   Runtime 测试保留 E2E 名称。
6. 所有仅验证 Mock 调用、“没有抛异常”或项目私有字段的测试改为断言公开输出、状态和
   可观察副作用。
7. `live_model` 测试使用 `PRAXIS_MODEL_API_KEY` 临时环境变量验证指定私有端点；密钥不
   写入文件、命令输出或测试报告。

## 错误与可观测性

新增 `InputError` 族异常：`InvalidInputSourceError`、`InputPathError`、
`InputMediaTypeError`、`InputSizeLimitError`、`InputNetworkError` 和
`UnsupportedInputModalityError`。异常 `details` 只包含种类、清理后的文件名、限制与
模型别名，不包含字节、Base64、URL 凭据或 API Key。

审计和事件只记录附件数量、模态、MIME 与大小。健康检查报告模型声明的多模态能力以及
输入解析能力；默认禁用远程输入属于安全配置，不是运行时故障。

## 完成标准

- AST 门禁在项目自有 Python 文件中报告零个单下划线私有符号或访问。
- 文本、图片、音频、视频、文件均经过公开 SDK 的流式和非流式路径测试。
- 路径穿越、符号链接逃逸、SSRF、重定向、MIME、大小和清理均有失败回归测试。
- 模拟场景测试不再宣称端到端；真实 HTTP SDK E2E 测试通过。
- 指定模型端点完成所有模态的真实能力探测并记录结果。
- Ruff、Pyright strict、全量 Pytest 90% 分支覆盖、pip-audit、构建和干净 wheel 冒烟
  全部通过，且 Windows/Linux 无未回收任务或线程。
