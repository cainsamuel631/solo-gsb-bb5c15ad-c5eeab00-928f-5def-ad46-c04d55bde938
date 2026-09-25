# GPU Fair-Queuing Scheduler (FastAPI + SQLite)

多租户 GPU 作业**加权 Deficit Round-Robin（WDRR）公平排队 API**，内置**等待时间老化补偿**防止小作业长期饥饿，并在任何时刻**不越过集群 GPU 容量约束**。

- **语言/框架**：Python 3.11 + FastAPI + Uvicorn
- **存储**：SQLite（WAL 模式，零外部依赖，数据持久化到挂载卷）
- **可复现**：固定租户访问顺序（租户 id 升序）、租户内 FIFO、顺序作业 id、无随机数；相同输入必然得到相同选择顺序
- **可审计**：每笔 `AGING / QUANTUM / DISPATCH` 余额变动都写入 `ledger`

---

## 一条命令启动

需要本机已安装 Docker 与 Docker Compose 插件（Docker Desktop / `docker-compose-plugin`）。
镜像构建只需 `requirements.txt`，SQLite 数据库文件（含**租户/作业/批次/账本与 GPU
维护时段**）保存在命名卷 `scheduler-data`（容器内 `/data/scheduler.db`），重建容器不丢数据：

```bash
docker compose up --build -d        # 构建镜像并后台启动
docker compose logs -f scheduler    # 跟踪服务日志（Ctrl-C 仅退出查看）
docker compose ps                   # 查看容器与健康检查状态
```

服务监听 **`http://localhost:8000`**（容器内 8000，可在 `docker-compose.yml` 改宿主机端口）。

停止 / 清理：

```bash
docker compose down          # 停止，数据保留在卷 scheduler-data
docker compose down -v       # 停止并删除数据
```

健康检查：

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

交互式 API 文档（Swagger UI）：<http://localhost:8000/docs>，OpenAPI：<http://localhost:8000/openapi.json>。

---

## 调度模型

### 成本单位

作业成本 = 资源占用：

```
cost_gpu_min = gpus × duration_min        # GPU-分钟
```

每个租户持有同一单位的 **deficit 余额（balance）**。

### 加权 DRR（每次 `POST /scheduler/batch`）

1. 取所有含 QUEUED 作业的租户，按**租户 id 升序**固定排列。
2. 每个 DRR 轮次，每位“队首作业**同时**能放入当前空闲 GPU、未超过本租户并发
   GPU 上限、且加入其成本后不超过本租户滚动 24 小时 GPU-分钟预算（见下两节）”
   的租户获得 `BASE_QUANTUM_GPU_MIN × weight` 的 QUANTUM 信用。
3. 按固定顺序访问租户：当 `余额 ≥ 队首作业成本` **且** `空闲 GPU ≥ 作业 GPU 数`
   **且** `租户 RUNNING 占用 GPU + 作业 GPU 数 ≤ 租户上限`
   **且** `窗口内已派发成本 + 本批前序成本 + 作业成本 ≤ 租户预算`
   时，派发该作业并扣减余额；否则继续。
4. **容量永远不越界**：放不下的作业——无论是受**集群容量**、**租户上限**还是
   **滚动 24h 预算**阻塞（已有运行作业占满上限、队首作业自身超过上限、或窗口内
   预算额度不足）——本轮不获得任何
   QUANTUM——等待释放由老化补偿，也避免重复调用接口“刷”余额。各租户的资格彼此
   独立：被阻塞的租户被跳过，**其他符合条件的租户本轮照常调度**。
5. “放得下但暂时买不起”的租户可在**同一批次内连续多轮累积**直到买得起；
   批次结束时未促成任何派发的尾部授权自动回滚，保证空转调用不改变余额。

#### 批次轮询：一次快照 + 租户队首游标

`/scheduler/batch` 与 `/scheduler/preview` 在**同一笔批次事务内只做一次读取**，
之后全部在内存里推进，二者因此共用同一条轮询代码路径：

- 批次开始时一次性读出**各租户按 FIFO（`queued_at, id`）排列的全部 QUEUED
  作业**、每个作业的**依赖状态**（未完成前置集合、最后前置完成时刻）以及**各租户
  当前 RUNNING GPU 合计**；整个批次不再重复查询队首或依赖。
- 每个租户持有一个指向其 FIFO 列表的**队首游标**。轮次按租户 id 升序访问，
  **游标只在该租户的队首作业被真正派发之后才前进**；放不下 / 买不起 / 被依赖
  阻塞时游标原地不动。
- 游标前进暴露出**新队首**后，会针对新队首**重新执行**整套判定：依赖是否就绪、
  预测运行区间上的**维护预留峰值**、租户**并发 GPU 上限**、以及**滚动 24h
  GPU-分钟预算**；批次内作业只可能因被派发而离开 QUEUED（前置不会在此期间完成或
  取消），所以开批时算出的依赖就绪映射对所有轮次、所有新队首都持续有效。
- 老化结算、余额、`AGING / QUANTUM / DISPATCH` 账本事件与选择顺序在正式批次与
  预演间逐项一致；**尾部未促成任何派发的 QUANTUM 授权一律回滚**（预演则整笔事务
  回滚，零写入）。

### 租户并发 GPU 上限（可选 `max_gpus`）

每个租户可设置一个**并发 GPU 占用上限**（整数，正）；`null`（或旧租户未设置）
表示**不限额**，只受集群总容量约束：

- 通过 `POST /tenants` **设置 / 更新 / 清除**：
  - 请求体带 `"max_gpus": 4` → 设置为 4；
  - 请求体显式带 `"max_gpus": null` → 清除上限（恢复不限额）；
  - **完全省略** `max_gpus` 字段 → 保留该租户既有上限不变（只更新 `weight`）；
    新租户省略时默认为不限额。
- 提交作业（`POST /jobs`，含自动注册租户）**不会**改动既有上限；自动注册的
  新租户以及升级前已存在的旧租户一律默认为 `null`（不限额）。
- 调度时队首作业必须**同时**满足集群空闲容量与租户上限才会被派发；受上限阻塞
  的租户本轮**不获得 QUANTUM**（但与容量阻塞一样可正常获得老化补偿，不会饥饿），
  其余租户不受影响。
- 作业 **`complete` 或 `cancel`（RUNNING 中取消）立即释放**其占用的额度。
- 若队首作业自身 GPU 数大于租户上限，它会保持 QUEUED（提交时不拒绝，因为上限
  可随时调整）；调高或清除上限后下一批次即可调度。
- 上限**持久化在 SQLite** 的 `tenants.max_gpus` 列；对没有该列的旧数据库，启动时
  自动执行兼容迁移（`ALTER TABLE ADD COLUMN`），旧租户该列为 `NULL`（不限额）。
- 租户对象（`GET /tenants`、`GET /scheduler/state`、批次结果等）含
  `max_gpus`（上限，`null` 为不限）与 `running_gpus`（当前 RUNNING 作业占用 GPU
  数，即上限已用量）。

### 租户滚动 24 小时 GPU-分钟预算（可选 `budget_gpu_min`）

每个租户还可设置一个**滚动 24 小时 GPU-分钟预算**（正整数，单位与作业成本相同）；
`null`（或旧租户未设置）表示**不限额**：

- 批次以其参考时刻 `now` 为窗口终点，统计半开区间 **`(now − 24 小时, now]`**
  内该租户**已派发作业**的 `gpus × duration_min` 累计值（左开：恰好在
  `now − 24h` 派发的作业不计入；右闭：在 `now` 当刻派发的计入）。
- 派发候选作业前，准入条件在集群容量、维护时段与 `max_gpus` 之外再增加：

  ```
  窗口内历史已派发成本 + 本批次前序已选中作业成本 + 候选成本
      ≤ budget_gpu_min
  ```

  本批次前序选中的作业 `scheduled_at` 就是 `now`（落在窗口内），由同一个累计
  计数**只计一次**，不会重复扣减。
- 候选成本加入后**超额**时：该作业**保持 QUEUED**，且与容量/上限阻塞一样
  **本轮不获得 QUANTUM**（无法刷信用，老化补偿照常）；该租户被跳过，
  **其他租户本轮继续独立调度**。严格 FIFO 不变：被预算挡住的队首作业不跳过，
  其后即使有小额可负担作业也继续等待。
- 成本在**派发时一次性计入**；作业随后 `complete` 或（RUNNING 中）`cancel`
  **均不退还**。只有从未派发（一直 QUEUED，含排队中取消）的作业不计费。
- 随着时间推移，超过 24 小时的旧派发自动滑出窗口，剩余额度随之恢复。
- 通过 `POST /tenants` **设置 / 更新 / 清除**，约定与 `max_gpus` 完全一致：
  - 请求体带 `"budget_gpu_min": 1440` → 设置/更新；
  - 显式 `"budget_gpu_min": null` → 清除（恢复不限额）；
  - **完全省略**该字段 → 保留既有预算不变；新租户省略时默认不限额。
  - 0/负数/小数/布尔 → 422（服务层直接调用返回 400）。
- `POST /jobs` 自动注册租户**不会**改动既有预算；预算**持久化在 SQLite** 的
  `tenants.budget_gpu_min` 列；对没有该列的旧数据库，启动时自动
  `ALTER TABLE ADD COLUMN` 补列，旧租户为 `NULL`（不限额）。
- 租户查询（`GET /tenants`、`GET /tenants/{id}`、`GET /scheduler/state`）以
  **服务器当前 UTC 时刻**为窗口终点返回：
  - `budget_gpu_min`：预算上限（`null` = 不限额）；
  - `budget_used_gpu_min`：当前 24 小时窗口内已用 GPU-分钟；
  - `budget_remaining_gpu_min`：窗口内剩余额度（不限额时为 `null`）；
  - `budget_window_start_at` / `budget_window_end_at`：上述数值实际对应的
    `(now−24h, now]` 窗口端点（UTC）。批次结果中的同名字段以**该批次参考时刻**
    为窗口终点，与该批的决策严格自洽；`GET /queue` 的每租户分组也带这些字段。
- **预演 `/scheduler/preview` 与正式批次共用同一派发代码路径**：预算判定完全
  相同，但预演事务回滚，候选成本不落库、不产生任何计费记录，可任意重复调用。

### 老化补偿（防饥饿）

- 只补偿**真正可调度之后**的等待：队首作业被依赖阻塞时不算调度饥饿。批次结算时，
  信用从队首作业**真正就绪的时刻**起算——多个前置作业取**最后完成时刻**
  （`MAX(completed_at)`），若是**依赖修订**使阻塞作业就绪则取**修订时刻**（记录在
  作业的 `ready_since` 上），并与该租户**已有老化结算点取较晚值**。因此「队首排队
  60 分钟、前置仅在调度前 1 分钟完成、期间未运行批次」只发放 **1 分钟** AGING
  信用，而不是 60 分钟。
- 除上述就绪时刻外，结算起点仍为：首次观察取自租户最早排队作业的到达时刻，之后
  自上次结算点继续；每等待 1 分钟按 `AGING_RATE_GPU_MIN_PER_MIN` 发放 GPU-分钟
  信用（可用 `AGING_CAP_GPU_MIN` 限制单批次上限）。
- 效果：小作业即便排在高权重大租户后面，等待越久信用越高，最终必然能负担，
  从而对等待时间形成上界。
- 队列排空后老化时钟重置，新作业从自己的到达时刻重新计时。
- 同一时刻重复调度不会产生信用（不可刷）。

### 生命周期

```
QUEUED ──(batch 选中)──▶ RUNNING ──complete──▶ COMPLETED
   │                        │
   └──────cancel────────────┴──────cancel────▶ CANCELLED
```

RUNNING 作业占用 GPU，直到 `complete`/`cancel` 才释放给后续批次——释放的既包括
**集群容量**，也包括**租户并发 GPU 上限**（`max_gpus`）的占用额度。

### 作业依赖（`depends_on`，DAG）

提交作业时可传入前置作业 id 列表；依赖是**集群级**的，允许**跨租户**依赖：

- 前置作业**全部 `COMPLETED`** 后，依赖作业才可被批次调度。
- 提交时校验：前置作业不存在 → `400 unknown depends_on`；形成环路（含自依赖）
  → `400 dependency cycle`，作业不会创建。
- 批次中只有「队首作业的前置已全部完成」的租户参与 DRR；队首被依赖阻塞时
  **不获得 QUANTUM、也不产生老化信用**（仅推进老化计时点），因此空等依赖
  无法“刷”出余额；严格 FIFO，被阻塞队首后面的本租户作业同样等待。
- 阻塞解除后（最后一个前置完成，或依赖修订使其就绪），老化信用**只从该就绪
  时刻起算**，绝不回溯到提交时刻（见「老化补偿」一节）。
- 前置作业被 **`CANCELLED`** 后永远不可能 COMPLETED：依赖作业**保持 QUEUED**，
  通过 `GET /jobs/{id}`、`GET /jobs` 与 `GET /queue` 的 `blocked_by` 字段可查询
  阻塞原因（前置作业 id 及其当前状态）。
- 作业对象包含：
  - `depends_on`：全部前置作业 id（去重、按 id 排序）；
  - `blocked_by`：尚未 COMPLETED 的前置作业 `[{id, status}]`，仅对 QUEUED 作业有意义，空数组表示立即可调度。

### 批量作业图提交（`POST /jobs/graph`，单事务原子落库）

需要一次提交**多个互相依赖的作业**时，可使用批量作业图接口；整个请求在
**同一把进程写锁与同一笔 `BEGIN IMMEDIATE` 写事务**内完成全部校验与写入，
任一错误**整批回滚**，不会改动任何租户权重或队列。

- 请求体含一个**非空作业数组 `jobs`**，每项字段：

  | 字段 | 必填 | 说明 |
  |---|---|---|
  | `ref` | 是 | 本批唯一的非空客户端标签，供批内其他作业通过 `depends_on_refs` 引用；响应里返回每个 ref 分配到的作业 id |
  | `tenant_id` | 是 | 非空租户标识（自动注册/更新租户，规则同 `POST /jobs`，不改动既有 `max_gpus`/`budget_gpu_min`） |
  | `gpus` | 是 | 所需 GPU 数（正整数，超过 `CLUSTER_GPUS` 整批拒绝） |
  | `duration_min` | 是 | 预计时长（分钟，正整数） |
  | `weight` | 是 | 租户权重（> 0）；**同一租户在本批的多项 weight 必须完全一致，否则整批报错** |
  | `depends_on` | 否 | 引用**已存在作业 id** 的前置列表（跨租户允许），未知 id 整批拒绝 |
  | `depends_on_refs` | 否 | 引用**本批内其他项 `ref`** 的前置列表；**允许前向引用**（数组无需拓扑排序），引用不存在的 ref 整批拒绝 |

- **依赖语义与单作业提交完全一致**：两个列表合并后去重、按作业 id 排序落库；
  允许前向及跨租户引用；整图（已有边 + 本批全部新边）做一次**环路校验**，
  成环（含自依赖）整批拒绝。
- 校验全部通过后才分配顺序作业 id（**按数组顺序** `J000001, J000002, …`，
  也是失败回滚后不留 id 空洞的同一计数口径），随后一次性写入租户、作业与全部
  依赖边（先插入全部作业再写边，因此前向引用不会触发外键问题）。
- **新作业立刻就是普通 QUEUED 作业**：成功响应返回后，下一次
  `POST /scheduler/preview` 与 `POST /scheduler/batch` 即按现有依赖判定
  （`blocked_by`、队首阻塞不发 QUANTUM/老化等）对待它们，无任何额外刷新步骤。
- 整批失败（重复 ref、同租户 weight 不一致、超容量、未知外部 id / 批内 ref、
  成环等，均为 **400**；请求体结构错误如空数组/缺字段为 **422**）时：
  不创建任何租户、不更新任何租户权重（含 `updated_at`）、不创建作业或边、
  不改变队列，失败批次也**不占用作业 id**。

访问地址（本机）：**`http://localhost:8000/jobs/graph`**（Docker Compose 默认
把容器 8000 端口映射到本机 **8000**）。

```bash
# team-a 的 train 依赖批内前向 ref "prep" 与已存在作业 J000001（跨租户）；
# team-b 的 serve 依赖 train 与 prep
curl -X POST http://localhost:8000/jobs/graph \
  -H 'Content-Type: application/json' \
  -d '{
    "jobs": [
      {"ref": "train", "tenant_id": "team-a", "gpus": 2, "duration_min": 10,
       "weight": 1, "depends_on": ["J000001"], "depends_on_refs": ["prep"]},
      {"ref": "prep",  "tenant_id": "team-a", "gpus": 1, "duration_min": 5,
       "weight": 1},
      {"ref": "serve", "tenant_id": "team-b", "gpus": 1, "duration_min": 5,
       "weight": 2, "depends_on_refs": ["train", "prep"]}
    ]
  }'
```

```jsonc
// 201
{
  "ref_ids": {"train": "J000002", "prep": "J000003", "serve": "J000004"},
  "jobs": [
    {"id": "J000002", "tenant_id": "team-a", "status": "QUEUED",
     "depends_on": ["J000001", "J000003"],
     "blocked_by": [{"id": "J000001", "status": "QUEUED"},
                    {"id": "J000003", "status": "QUEUED"}], ...},
    {"id": "J000003", "tenant_id": "team-a", "status": "QUEUED",
     "depends_on": [], "blocked_by": [], ...},
    {"id": "J000004", "tenant_id": "team-b", "status": "QUEUED",
     "depends_on": ["J000002", "J000003"], "blocked_by": [...], ...}
  ]
}
```

同租户 weight 不一致会被整批拒绝（且不改动该租户既有权重）：

```bash
curl -X POST http://localhost:8000/jobs/graph \
  -H 'Content-Type: application/json' \
  -d '{"jobs": [
        {"ref":"a","tenant_id":"team-a","gpus":1,"duration_min":5,"weight":1},
        {"ref":"b","tenant_id":"team-a","gpus":1,"duration_min":5,"weight":2}
      ]}'
# 400 {"error":"tenant 'team-a' has inconsistent weights in the same graph:
#       1.0 vs 2.0 (ref 'b')"}
```

### 排队作业的依赖修订（`PATCH /jobs/{id}/dependencies`）

已入队（**QUEUED**）作业的前置依赖可整体修订：请求体的 `depends_on` 数组
**替换**现有全部前置（空数组 `[]` 解除所有依赖），自动去重并按作业 id 排序，
**允许跨租户引用**：

- 校验与提交时一致：未知前置 id、自依赖、替换后成环一律拒绝（400），且
  **原依赖保持不变**（单次写事务整体回滚，不会留下改了一半的边）。
- **仅 QUEUED 可修订**：RUNNING/COMPLETED/CANCELLED → 409（已派发作业的前置
  是历史事实）；作业不存在 → 404。
- 修订与批次调度**共用同一进程写锁与 SQLite 写事务**（`BEGIN IMMEDIATE`），
  对并发批次/预演原子生效；成功后的**下一次预演与正式批次即按新依赖判断**。
- 作业在队列中的位置（`queued_at`）与租户余额/老化结算点均不受影响；唯一更新的
  是作业自身的**就绪时钟** `ready_since`：修订使**阻塞作业就绪**时记录修订时刻
  （老化自该时刻起算），修订使其**重新阻塞**时清零，**就绪→就绪**时保留原就绪
  时刻不变。
- 修订**不改写首次提交的幂等指纹**：用原 `Idempotency-Key` + 原参数重试
  `POST /jobs` 只会返回（可能已修订的）作业当前对象，不会把依赖改回提交时的样子。

### 依赖取消级联清理（`cancel-impact` / `cancel-cascade`）

取消一个作业后，所有（传递）依赖它的排队作业将永远等不到前置完成。本模块沿
**跨租户依赖图**求传递闭包，预览并清理这些“永久受阻”的排队后代：

- **闭包遍历规则**：从目标作业出发沿「前置 → 依赖者」方向广度优先遍历；
  **穿过已取消（CANCELLED）节点**——已取消的前置永远不可能完成，阻塞会透过它
  继续传播；**不穿过已完成（COMPLETED）节点**——已满足的前置不阻塞任何人，
  闭包到此为止。
- `GET /jobs/{id}/cancel-impact`（**纯预览，不写库**）：返回目标作业信息及
  **按作业 id 排序**的受影响 QUEUED 作业列表，每个作业附带**一条最短依赖链**
  （从目标到该作业，链上每个作业依赖前一个；等长时取 id 升序 BFS 首先到达者，
  结果确定）。
- `POST /jobs/{id}/cancel-cascade`（**执行清理**）：在与现有取消、依赖修订
  **共用的进程写锁与 `BEGIN IMMEDIATE` 写事务**内**重新计算**影响范围，原子地
  取消目标及范围内仍为 QUEUED 的全部作业；目标若为 RUNNING，其占用的 GPU
  （含租户 `max_gpus` 额度）立即释放给后续批次。**已取消的中间节点只用于遍历，
  不会被重复修改**（其 `completed_at` 等保持不变）；队列排空的租户老化时钟随之
  重置，与单个取消一致。
- 两个接口对目标的校验一致：不存在 → 404；已完成或已取消 → 409。
- 执行后现有作业与队列接口（`GET /jobs`、`GET /queue`、批次调度等）立即反映
  结果：被级联取消的作业为 `CANCELLED`，不再参与调度。

### GPU 维护时段（`maintenance_windows`，持久化）

可创建**持久化的 GPU 维护时段**：每段是一个 **UTC 半开区间 `[start_at, end_at)`**，
并声明该时段内为维护**预留的 GPU 数**（`reserved_gpus`）。维护时段只影响**后续派发**，
**已在 RUNNING 的作业永不被抢占**——新时段不会取消或挪动运行中的作业。

- 批次以其参考时刻 `now` 作为该批每个被选作业的**预计开始时刻**，结合作业既有的
  `duration_min` 形成预计运行的**半开区间 `[now, now + duration_min)`**。
- 每次派发候选作业前，计算与其预计区间**重叠的维护时段在该区间内产生的最大（峰值）
  预留量**：
  - 同一时刻重叠的多个时段，预留量**相加**；
  - 时段**端点相接不算重叠**（半开：作业恰好在某时段开始时结束、或在某时段结束时
    开始，都不计入），因此相接的两个时段预留量不累加；
  - 取整个预计区间内的**峰值合计预留量** `R`（不是各时段预留量的简单求和）。
- 准入条件（在既有集群容量与租户 `max_gpus` 判定之外新增）：

  ```
  已有 RUNNING 占用 + 同批次已选占用 + 候选作业占用
      ≤ 集群容量 − 该候选区间的峰值预留量 R
  ```

  因此短作业（在时段开始前就结束）可以照常派发，而会跨越时段的长作业必须等待；
  时段结束后，用更晚的 `now` 再跑批次即可派发。
- **预演与正式批次共用同一条判定代码路径**（同一事务体，仅提交/回滚之别）：相同
  `now` 与相同状态下选择顺序完全一致；预演不落库，已持久化的维护时段保持不变。
- 校验：`end_at <= start_at` → 400；`reserved_gpus` 超出 `0..CLUSTER_GPUS` → 400
  （负整数/布尔等类型错误为 422）。允许预留 `0`（相当于占位、不影响调度）。
- **累计预留校验**：创建时在**同一写事务**内把候选时段与已持久化时段按同一套 UTC
  半开区间扫描叠加，计算候选覆盖期间 `[start_at, end_at)` 内「已有时段 + 候选时段」
  的**预留峰值**；峰值**超过当前集群容量** → 400 且**不落库**（也不占用 id）。
  与调度口径一致：**端点相接不累加**，峰值**恰好等于容量可以创建**。例如 8 GPU
  集群先建 `[10:00,12:00)` 预留 5，再建 `[11:00,13:00)` 预留 4 会因重叠段
  `[11:00,12:00)` 峰值 9 > 8 返回 400；改预留 3（峰值恰为 8）或建相接时段
  `[12:00,14:00)` 预留 8 均可创建。
- 维护时段与作业、租户、账本一样保存在 SQLite（`maintenance_windows` 表），随现有
  Docker Compose 的命名卷 `scheduler-data`（容器内 `/data/scheduler.db`）持久化，
  重建容器不丢；旧数据库启动时自动建表，无需手工迁移。
- 批次结果中的相关字段：
  - 顶层 `maintenance_reserved_gpus_at_start`：批次参考时刻正处于活动状态的预留合计，
    `effective_free_gpus_at_start`：扣除它之后的即时可用 GPU（不计已有 RUNNING 之外的
    未来区间影响）；
  - 每个 `selected_jobs` 条目附带 `estimated_start_at` / `estimated_end_at`
    （预计半开运行区间）与 `max_maintenance_reserved_gpus`（该区间峰值预留量）。
- 对仍在排队的作业，可用 `GET /jobs/{id}/maintenance-slots` 直接查询它在指定
  时间范围内**可行的预计开始时刻区间与最早时刻**（见下文「6b」节）。

---

## 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `CLUSTER_GPUS` | `8` | 集群总 GPU 数，任何批次不会超过 |
| `BASE_QUANTUM_GPU_MIN` | `60` | 每单位权重每轮的量子（GPU-分钟） |
| `AGING_RATE_GPU_MIN_PER_MIN` | `1` | 每等待 1 分钟补偿的 GPU-分钟 |
| `AGING_CAP_GPU_MIN` | 空（无上限） | 单批次老化补偿上限 |
| `DB_PATH` | `/data/scheduler.db` | SQLite 文件路径 |

提交作业时若 `gpus > CLUSTER_GPUS`（永远无法运行）会被拒绝（400）。无
`Idempotency-Key` 的提交始终按当前容量校验；带 key 时只有**未命中的首次提交**
按当前容量校验——命中已有 key 的重试先比对已保存指纹（同参 200、异参 409），
容量下调不会让同参重试变成 400（见下文「幂等提交」一节）。

> 注意：租户的并发 GPU 上限 `max_gpus` 与滚动 24 小时 GPU-分钟预算
> `budget_gpu_min` **都不是**环境变量，而是通过接口逐租户设置并持久化在
> SQLite 中；环境变量只配置上面这些全局参数。

---

## API 用法

基础地址：`http://localhost:8000`

### 0. 管理租户（权重、并发 GPU 上限与 24 小时 GPU-分钟预算）`POST /tenants`

```bash
# 创建租户：权重 2，并发 GPU 上限 4，滚动 24h GPU-分钟预算 1440
curl -X POST http://localhost:8000/tenants \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","weight":2.0,"max_gpus":4,"budget_gpu_min":1440}'

# 更新权重但保留原上限与预算（省略两个可选字段）
curl -X POST http://localhost:8000/tenants \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","weight":1.5}'

# 清除预算（显式 null => 恢复不限额）；未提供的 max_gpus 同理会保留
curl -X POST http://localhost:8000/tenants \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","weight":1.5,"budget_gpu_min":null}'
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `tenant_id` | 是 | 非空租户标识 |
| `weight` | 是 | 租户权重，必须 > 0 |
| `max_gpus` | 否 | 并发 GPU 上限，正整数；显式 `null` 清除上限；**省略**时保留既有上限（新租户为不限额 `null`）。0/负数 → 422 |
| `budget_gpu_min` | 否 | **滚动 24 小时 GPU-分钟预算**，正整数；显式 `null` 清除（不限额）；**省略**时保留既有预算（新租户为不限额 `null`）。0/负数/小数/布尔 → 422 |

响应为租户对象，除权重/余额外还包含：

- `max_gpus`：并发 GPU 上限（`null` = 不限额）；
- `running_gpus`：该租户当前 RUNNING 作业合计占用的 GPU 数（上限已用量）；
- `budget_gpu_min`：滚动 24h GPU-分钟预算上限（`null` = 不限额）；
- `budget_used_gpu_min`：以**服务器当前 UTC 时刻**为终点的 `(now−24h, now]`
  窗口内已派发作业成本合计（完成/取消不退还）；
- `budget_remaining_gpu_min`：同一窗口内剩余可派发额度（不限额时为 `null`）；
- `budget_window_start_at` / `budget_window_end_at`：上述窗口的 UTC 起止端点；
- `queued_jobs` / `running_jobs`：队列中 / 运行中的作业数。

### 1. 提交作业 `POST /jobs`

访问地址（本机）：**`http://localhost:8000/jobs`**，端口 **`8000`**
（Docker Compose 映射，可在 `docker-compose.yml` 修改宿主机端口）。

提交作业时租户会被自动注册（也可用 `POST /tenants` 单独管理权重、`max_gpus`
与 `budget_gpu_min`）。同一租户重复提交可更新其权重，但**不会**改动该租户已
设置的 `max_gpus` 上限或 `budget_gpu_min` 预算（自动注册的新租户二者均为
`null`，即不限额/不限预算）。

```bash
curl -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","gpus":2,"duration_min":30,"weight":1}'
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `tenant_id` | 是 | 非空租户标识 |
| `gpus` | 是 | 所需 GPU 数（正整数） |
| `duration_min` | 是 | 预计时长（分钟，正整数） |
| `weight` | 否 | 租户权重，默认 `1.0`，必须 > 0 |
| `depends_on` | 否 | 前置作业 id 列表（默认 `[]`），**跨租户允许**；未知 id 或形成环路返回 400，自动去重 |

响应：作业对象（顺序 id 如 `J000001`，`status: QUEUED`），含 `depends_on` 与
`blocked_by` 字段（见上文「作业依赖」一节）。

#### 可选请求头：幂等提交 `Idempotency-Key`

`POST /jobs` 支持**可选**请求头 `Idempotency-Key`（任意非空字符串，通常由客户端
生成，如 UUID）。幂等域为 **`tenant_id` + 非空 key**：同一租户使用相同 key 的
多次请求只代表一次提交；不同租户使用相同 key 互不影响。

- **判定顺序（重要）**：带 key 的请求进入写事务后，**先比对该
  `(tenant_id, key)` 已保存的请求指纹，再做任何容量判定或写入**：
  1. **命中已保存凭据**：指纹相同 → 返回该作业的**当前对象**（可能已被
     派发/完成/取消，HTTP **200**），**不再更新**租户权重（含
     `updated_at`）、**不改变排队时间** `queued_at`、**不重复写入依赖**、
     不新建第二个作业，并且**完全不再按当前容量重新校验**——即使服务重启时
     `CLUSTER_GPUS` 被下调、使原请求现在已经超过现有容量，重试依旧返回该
     作业当前状态与 200；指纹不同（参数被改动）→ **409**，且**与当前容量
     无关**——哪怕新参数本身就超过当前容量，也是 409。
  2. **未命中（该 key 的首次提交）**：此时才按**当前** `CLUSTER_GPUS`
     做容量校验，通过才创建并落库指纹；任一失败（超容量、未知依赖、依赖环
     等，无论发生在事务开启前还是事务内）整体回滚，**失败不占用 key**，用
     同一 key 修正参数后重试仍会创建作业。
- **无 key（或值为空白）提交完全沿用原接口**：每次请求都新建作业、返回 201，
  容量校验照旧始终按当前 `CLUSTER_GPUS` 执行（重启后容量下调，无 key 的
  超容量请求仍返回 400）。
- **并发与重启**：同一 (tenant, key) 的并发首次请求——无论是同进程多线程、多个
  容器进程共享同一个 SQLite 文件，还是客户端/容器重启后重试——都**只会创建一个
  作业**（进程写锁 + `BEGIN IMMEDIATE` 串行化，数据库 `PRIMARY KEY
  (tenant_id, idempotency_key)` 唯一约束兜底，凭据随命名卷持久化）。重启后重试
  绝不改写租户、作业或账本。

```bash
# 首次提交 201；网络重试时用完全相同的请求体与同一个 key
curl -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: 7e2f9b2c-1d4a-4e6b-9c8f-0a1b2c3d4e5f' \
  -d '{"tenant_id":"team-a","gpus":2,"duration_min":30,"weight":1,
       "depends_on":["J000003","J000001","J000003"]}'
# 201 {"id":"J000004", ..., "depends_on":["J000001","J000003"], ...}

# 相同 key + 相同参数（依赖顺序/重复与否无关，按去重排序后的指纹比较）-> 200，
# 返回 J000004 的当前对象，不产生任何副作用
curl -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: 7e2f9b2c-1d4a-4e6b-9c8f-0a1b2c3d4e5f' \
  -d '{"tenant_id":"team-a","gpus":2,"duration_min":30,"weight":1,
       "depends_on":["J000001","J000003"]}'
# 200 {"id":"J000004", ...}

# 相同 key 但参数不同 -> 409
curl -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: 7e2f9b2c-1d4a-4e6b-9c8f-0a1b2c3d4e5f' \
  -d '{"tenant_id":"team-a","gpus":4,"duration_min":30}'
# 409 {"error":"Idempotency-Key '...' for tenant 'team-a' was already used
#       with different parameters; existing job: J000004"}
```

重启后容量下调（如把 docker-compose.yml 的 `CLUSTER_GPUS` 由 `8` 改为 `4` 并
重建容器）时，重试规则依旧只看已保存指纹，不按新容量二次校验：

```bash
# 容量为 8 时首次提交 6 GPU 作业 -> 201
curl -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: retry-after-shrink' \
  -d '{"tenant_id":"team-a","gpus":6,"duration_min":30}'

# …随后 CLUSTER_GPUS 下调为 4，服务重启…

# 相同 key + 相同参数（现在已超过当前容量 4）-> 仍是 200，返回作业当前对象
curl -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: retry-after-shrink' \
  -d '{"tenant_id":"team-a","gpus":6,"duration_min":30}'
# 200 {"id":"J....", ...}   （作业可能已 RUNNING/COMPLETED/CANCELLED）

# 相同 key + 不同参数（无论 5 GPU 超容量，还是 2 GPU 不超容量）-> 一律 409
# 新 key 的首次提交才按当前容量校验：6 GPU -> 400，且失败不占用该新 key；
# 改成 4 GPU 后用同一新 key 重试 -> 201
```

依赖提交示例（`J000002` 等 `J000001` 完成后才可调度）：

```bash
curl -X POST http://localhost:8000/jobs \
  -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-b","gpus":1,"duration_min":30,"depends_on":["J000001"]}'
# 201 {"id":"J000002","status":"QUEUED",
#      "depends_on":["J000001"],
#      "blocked_by":[{"id":"J000001","status":"QUEUED"}], ...}
```

### 1b. 修订排队作业的依赖 `PATCH /jobs/{id}/dependencies`

整体替换一个 **QUEUED** 作业的前置依赖集（跨租户允许，自动去重并按 id 排序）：

```bash
# J000005 改为等待 J000001 与 J000002
curl -X PATCH http://localhost:8000/jobs/J000005/dependencies \
  -H 'Content-Type: application/json' \
  -d '{"depends_on":["J000002","J000001","J000002"]}'
# 200 {"id":"J000005","status":"QUEUED",
#      "depends_on":["J000001","J000002"],
#      "blocked_by":[{"id":"J000001","status":"RUNNING"},
#                    {"id":"J000002","status":"QUEUED"}], ...}

# 空数组解除全部依赖（下一次预演/批次即可参与调度）
curl -X PATCH http://localhost:8000/jobs/J000005/dependencies \
  -H 'Content-Type: application/json' \
  -d '{"depends_on":[]}'
# 200 {"id":"J000005","status":"QUEUED","depends_on":[],"blocked_by":[], ...}
```

| 情形 | 状态码 | 说明 |
|---|---|---|
| 成功 | 200 | 返回含最新 `depends_on` / `blocked_by` 的作业对象 |
| 作业不存在 | 404 | — |
| 作业非 QUEUED | 409 | RUNNING/COMPLETED/CANCELLED 的前置已是历史事实 |
| 未知前置 id / 自依赖 / 替换后成环 | 400 | **原依赖保持不变**（单次写事务整体回滚） |
| `depends_on` 缺失 / 非数组 / 含空白字符串 | 422 | 请求体校验失败 |

修订与批次调度共用同一写锁与 `BEGIN IMMEDIATE` 写事务：修订结果对随后的
`POST /scheduler/preview` 与 `POST /scheduler/batch` 立即可见。修订**不会**改写
首次提交的幂等指纹：用原 `Idempotency-Key` + 原参数重试 `POST /jobs` 只会返回
（可能已修订的）作业当前对象，不会把依赖改回提交时的样子。

### 2. 调度下一批 `POST /scheduler/batch`

执行一次 WDRR + 老化批次（**会落库**：作业转 RUNNING、扣减余额、写账本与批次、
按 `scheduled_at = now` 计入租户滚动 24h GPU-分钟预算），
返回**选择顺序**、每租户**余额**、容量使用与完整事件流水：

```bash
curl -X POST http://localhost:8000/scheduler/batch
# 请求体可省略（沿用服务器当前时刻作为参考时刻）；也可显式传入参考时刻：
curl -X POST http://localhost:8000/scheduler/batch \
  -H 'Content-Type: application/json' \
  -d '{"now":"2026-09-23T12:00:00Z"}'
```

```jsonc
{
  "batch_id": 1,
  "capacity_gpus": 8,
  "used_gpus_before": 0,
  "available_gpus_at_start": 8,
  "used_gpus_after": 8,
  "free_gpus_after": 0,
  "rounds": 3,
  "jobs_selected": 2,
  "selection_order": ["J000001", "J000002"],
  "selected_jobs": [ { "id": "J000001", "run_index": 1, ... } ],
  "tenants": [
    {"id":"team-a","weight":1.0,"balance_gpu_min":0.0,"max_gpus":null,"running_gpus":1,"queued_jobs":0,"running_jobs":1},
    {"id":"team-b","weight":2.0,"balance_gpu_min":60.0,"max_gpus":null,"running_gpus":1,"queued_jobs":0,"running_jobs":1}
  ],
  "events": [
    {"kind":"AGING",  "tenant_id":"team-a", "amount":0.01, ...},
    {"kind":"QUANTUM","tenant_id":"team-a", "amount":60.0, "detail":"round 1"},
    {"kind":"DISPATCH","tenant_id":"team-a","amount":-60.0,"job_id":"J000001"}
  ]
}
```

### 2b. 预演下一批（dry-run，零写入）`POST /scheduler/preview`

在**不写入任何数据**的前提下预演一次批次，返回与正式批次相同结构的结果：
**预计派发顺序**、批次后的 **GPU 占用**、各租户**余额**与完整**信用事件**
（`AGING / QUANTUM / DISPATCH`）。

```bash
curl -X POST http://localhost:8000/scheduler/preview
# 或带可选参考时刻（UTC ISO-8601），便于把预演与随后正式批次对齐到同一时刻：
curl -X POST http://localhost:8000/scheduler/preview \
  -H 'Content-Type: application/json' \
  -d '{"now":"2026-09-23T12:00:00Z"}'
```

预演与 `POST /scheduler/batch` **共用同一套代码路径**——同一笔数据库事务内执行
完全相同的 WDRR、老化结算、租户并发上限与依赖判定，唯一区别是事务结尾 **回滚**
而非提交，因此两套算法不可能分叉：

- **零写入**：作业保持 QUEUED、余额与老化计时点不变、`batches`/`ledger` 不新增行；
  预演过程中**异常也会整体回滚**，绝不留下部分变更。
- **结果可复现一致**：同一数据库状态、相同参考时刻（`now`）下，预演结果与随后
  正式批次的决策逐项一致（派发顺序、`used/free_gpus_after`、`rounds`、事件金额与
  余额、租户快照）；响应仅多出 `"preview": true` 标记。回滚的批次/账本自增行会被
  随后的正式批次原样复用，因此连 `batch_id` 与事件 `id` 都一致，也不会产生 id 空洞。
- 响应中的 `batch_id`/事件 `id` 为**预演用的预期值（prospective）**，并不持久化；
  预演里 `selected_jobs` 的 `status: RUNNING`、`scheduled_at`、`batch_id` 同样只是
  “若正式执行会是什么样”的预测视图。
- 预演可**任意重复调用**：同刻重复预演结果一致，无法借预演“刷”量子或老化信用。
- 空队列预演返回 `selection_order: []`、`events: []`，同样零写入。

```jsonc
// POST /scheduler/preview 的响应（与正式批次同构，仅多 preview 标记）
{
  "batch_id": 1,
  "preview": true,
  "capacity_gpus": 8,
  "used_gpus_before": 0,
  "used_gpus_after": 8,
  "free_gpus_after": 0,
  "rounds": 3,
  "selection_order": ["J000001", "J000002"],
  "selected_jobs": [ ... ],   // 预测视图：status 显示为 RUNNING，但库里仍为 QUEUED
  "tenants": [ ... ],         // 预测的批次后余额/占用
  "events": [ ... ]           // 预测的 AGING/QUANTUM/DISPATCH 流水（不入库）
}
```

典型用法（先看后做，且结果对得上）：

```bash
curl -s -X POST localhost:8000/scheduler/preview \
  -H 'Content-Type: application/json' -d '{"now":"2026-09-23T12:00:00Z"}'
# 确认派发顺序/余额/占用符合预期后，用同一参考时刻正式执行：
curl -s -X POST localhost:8000/scheduler/batch \
  -H 'Content-Type: application/json' -d '{"now":"2026-09-23T12:00:00Z"}'
```

### 3. 完成 / 取消作业

```bash
curl -X POST http://localhost:8000/jobs/J000001/complete   # RUNNING -> COMPLETED，释放 GPU
curl -X POST http://localhost:8000/jobs/J000002/cancel     # QUEUED/RUNNING -> CANCELLED
```

- 只有 RUNNING 可 complete（否则 409）；已结束作业重复取消 409；不存在 404。

### 3b. 依赖取消级联：预览 `GET /jobs/{id}/cancel-impact`，执行 `POST /jobs/{id}/cancel-cascade`

预览取消某个 **QUEUED 或 RUNNING** 作业后会**永久受阻的排队后代**（沿跨租户
依赖图求传递闭包，穿过已取消节点、不穿过已完成节点；**纯读取，不写库**）：

```bash
# J000002 依赖 J000001，J000003 依赖 J000002（可跨租户）
curl http://localhost:8000/jobs/J000001/cancel-impact
```

```jsonc
{
  "job_id": "J000001",
  "status": "QUEUED",
  "affected_count": 2,
  "affected_jobs": [
    {"id": "J000002", "tenant_id": "team-b", "status": "QUEUED",
     "depends_on": ["J000001"], "chain": ["J000001", "J000002"], ...},
    {"id": "J000003", "tenant_id": "team-c", "status": "QUEUED",
     "depends_on": ["J000002"], "chain": ["J000001", "J000002", "J000003"], ...}
  ]
}
```

- `affected_jobs` **按作业 id 排序**，每个元素是完整作业对象并附带 `chain`：
  从目标到该作业的**一条最短依赖链**（链上每个作业依赖前一个）。

确认影响范围后执行级联取消（与现有取消、依赖修订共用同一写锁与 SQLite 写
事务，在执行事务内**重新计算**影响范围）：

```bash
curl -X POST http://localhost:8000/jobs/J000001/cancel-cascade
```

```jsonc
{
  "job_id": "J000001",
  "cancelled_count": 3,
  "cancelled": ["J000001", "J000002", "J000003"],
  "jobs": [ {"id": "J000001", "status": "CANCELLED", ...}, ... ]
}
```

- 目标及范围内仍为 QUEUED 的作业被原子取消；目标若为 RUNNING，其 GPU 占用
  立即释放给后续批次；**已取消的中间节点只用于遍历，不被重复修改**。
- 目标不存在 → 404；目标已完成或已取消 → 409（预览接口同样）。
- 执行后 `GET /jobs`、`GET /queue` 等接口立即反映：被清理的作业为
  `CANCELLED`，不再出现在队列中。

### 4. 查询队列 / 作业 / 租户 / 流水

```bash
curl http://localhost:8000/queue                    # 每租户 FIFO 队列、位置、权重、余额
curl "http://localhost:8000/jobs?status=QUEUED"     # 可按 status / tenant_id 过滤
curl http://localhost:8000/jobs/J000001
curl http://localhost:8000/tenants                  # 全部租户 + 余额
curl http://localhost:8000/tenants/team-a
curl http://localhost:8000/scheduler/state          # 容量、空闲 GPU、调参、租户余额
curl "http://localhost:8000/ledger?tenant_id=team-a&limit=200"  # AGING/QUANTUM/DISPATCH 审计
```

`GET /queue` 示例（`J000005` 等待 `J000003` 完成，队列中直接标出阻塞原因）：

```jsonc
{
  "cluster_gpus": 8, "used_gpus": 6, "free_gpus": 2,
  "queued_jobs": 1, "running_jobs": 1,
  "queues": [{
    "tenant_id": "team-c", "weight": 1.0, "balance_gpu_min": 12.5,
    "max_gpus": null,
    "budget_gpu_min": null, "budget_used_gpu_min": 0,
    "budget_remaining_gpu_min": null,
    "budget_window_start_at": "2026-09-21T10:00:00+00:00",
    "budget_window_end_at": "2026-09-22T10:00:00+00:00",
    "running_gpus": 0,
    "oldest_queued_at": "2026-09-22T10:00:00+00:00",
    "jobs": [{
      "id": "J000005", "position": 1, "gpus": 4, "status": "QUEUED",
      "depends_on": ["J000003"],
      "blocked_by": [{"id": "J000003", "status": "RUNNING"}], ...
    }]
  }]
}
```

### 5. 按时间窗汇总 GPU 用量 `GET /usage/gpu`

按 UTC 时间窗 `[start, end)`（**左闭右开**）汇总每租户与全局的实际 GPU 用量：

- 用量来自作业的**实际 RUNNING 区间**，而非提交时的 `duration_min` 估算：
  - **已结束作业**（`COMPLETED`，或运行中被 `CANCELLED`）：`scheduled_at → completed_at`；
  - **仍在运行的作业**：`scheduled_at` 起算，**截断到查询截止时刻 `end`**（不假设作业在未来仍持续）。
- 与查询区间按左闭右开求交：作业恰好在 `end` 才启动、或恰好在 `start` 结束，用量均为 **0**。
- **QUEUED 作业不计入**（从未占用 GPU）；排队期间被取消的作业同样不计。
- 用量 = `gpus × 交集分钟数`；`overlapping_jobs` 为 RUNNING 区间与时间窗有非空交集的作业数（不是 GPU 数）。
- 返回所有已知租户（按 id 排序），**无重叠作业的租户数值为零**；时间窗内无任何作业时，全局用量也为零。
- 查询参数：

| 参数 | 必填 | 说明 |
|---|---|---|
| `start` | 是 | 起始时刻（**UTC**，ISO-8601，如 `2026-09-23T00:00:00Z`），含 |
| `end` | 是 | 截止时刻（**UTC**，ISO-8601），不含；**必须晚于 `start`，否则 400** |
| `tenant_id` | 否 | 只统计单个租户（未知租户返回 404）；缺省统计全部租户 |

```bash
curl -G "http://localhost:8000/usage/gpu" \
  --data-urlencode "start=2026-09-23T00:00:00Z" \
  --data-urlencode "end=2026-09-23T01:00:00Z"
# 只看某租户：
curl -G "http://localhost:8000/usage/gpu" \
  --data-urlencode "start=2026-09-23T00:00:00Z" \
  --data-urlencode "end=2026-09-23T01:00:00Z" \
  --data-urlencode "tenant_id=team-a"
```

```jsonc
{
  "start": "2026-09-23T00:00:00+00:00",
  "end": "2026-09-23T01:00:00+00:00",
  "tenant_id": null,
  "global": { "gpu_minutes": 180.0, "overlapping_jobs": 2 },
  "tenants": [
    {"tenant_id": "team-a", "gpu_minutes": 60.0,  "overlapping_jobs": 1},
    {"tenant_id": "team-b", "gpu_minutes": 120.0, "overlapping_jobs": 1}
  ]
}
```

### 6. GPU 维护时段 `POST` / `GET /maintenance-windows`

创建一个持久化的维护时段（UTC 半开区间 `[start_at, end_at)`，预留 `reserved_gpus`
块 GPU）：

```bash
curl -X POST http://localhost:8000/maintenance-windows \
  -H 'Content-Type: application/json' \
  -d '{"start_at":"2026-09-24T02:00:00Z","end_at":"2026-09-24T04:00:00Z","reserved_gpus":4}'
# 201
# {"id":1,"start_at":"2026-09-24T02:00:00+00:00","end_at":"2026-09-24T04:00:00+00:00",
#  "reserved_gpus":4,"created_at":"..."}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `start_at` | 是 | 起始时刻（**UTC**，ISO-8601），含；不带时区偏移的时间戳按 UTC 解释 |
| `end_at` | 是 | 结束时刻（**UTC**，ISO-8601），不含；**必须严格晚于 `start_at`，否则 400** |
| `reserved_gpus` | 是 | 时段内预留 GPU 数，整数 `0..CLUSTER_GPUS`；单条超出集群容量 → 400，负数/布尔 → 422。另做**累计校验**：候选覆盖期间「已有时段 + 候选时段」的预留峰值超过当前容量同样 → 400 且不落库（端点相接不累加，峰值等于容量可创建） |

查询全部维护时段（按开始时刻升序）：

```bash
curl http://localhost:8000/maintenance-windows
# 只看某一 UTC 时刻处于活动状态的时段（同样半开：end_at 恰好等于该时刻不算活动）：
curl -G http://localhost:8000/maintenance-windows \
  --data-urlencode "active_at=2026-09-24T03:00:00Z"
```

维护时段如何影响批次（示意，集群 8 GPU）：

```bash
# 8-GPU 作业预计运行 [00:00, 01:00)；00:30 起预留 2 块 -> 该作业整段不能派发
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"a","gpus":8,"duration_min":60}'
curl -s -X POST localhost:8000/maintenance-windows -H 'Content-Type: application/json' \
  -d '{"start_at":"2026-01-01T00:30:00Z","end_at":"2026-01-01T01:30:00Z","reserved_gpus":2}'
curl -s -X POST localhost:8000/scheduler/preview \
  -H 'Content-Type: application/json' -d '{"now":"2026-01-01T00:00:00Z"}'
# {"selection_order": [], "maintenance_reserved_gpus_at_start": 0, ...}

# 端点相接不算重叠：作业 [00:00,01:00) 与时段 [01:00,02:00) 不相交，照常派发
curl -s -X POST localhost:8000/maintenance-windows -H 'Content-Type: application/json' \
  -d '{"start_at":"2026-01-02T01:00:00Z","end_at":"2026-01-02T02:00:00Z","reserved_gpus":8}'
curl -s -X POST localhost:8000/scheduler/batch \
  -H 'Content-Type: application/json' -d '{"now":"2026-01-02T00:00:00Z"}'
# {"selection_order": ["J000001"],
#  "selected_jobs":[{"id":"J000001","estimated_start_at":"2026-01-02T00:00:00+00:00",
#                   "estimated_end_at":"2026-01-02T01:00:00+00:00",
#                   "max_maintenance_reserved_gpus":0, ...}], ...}
```

### 6b. 排队作业维护空档查询 `GET /jobs/{id}/maintenance-slots`

对一个 **QUEUED** 作业，查询它在指定 UTC **半开区间 `[start, end)`** 内所有
**可行的预计开始时刻**，以及最早可行时刻：

- 以查询时刻**当前 RUNNING 作业占用的 GPU 数为固定基线**，在整个查询视野内
  保持不变——**不推测**运行中作业未来何时完成释放。
- 对候选开始时刻 `t`，把作业**整段预计运行区间 `[t, t + duration_min)`** 与
  已保存的维护时段叠加：任一时刻
  `RUNNING 基线 + 作业 GPU 数 + 该时刻叠加预留量 > 集群容量`，
  该开始时刻即不可行。重叠时段预留量相加、**端点相接不算重叠**（与批次准入
  完全同一套数据与半开语义）。
- 结果**仅代表资源容量**：不参考租户上限/预算/余额/依赖，也不考虑其他排队作业。

查询参数（UTC ISO-8601；不带时区偏移的时间戳按 UTC 解释）：

| 参数 | 必填 | 说明 |
|---|---|---|
| `start` | 是 | 查询起点（**UTC**），含 |
| `end` | 是 | 查询终点（**UTC**），不含；**必须晚于 `start`，否则 400** |

响应字段：

- `slots`：可行开始时刻区间列表（按时间升序、重叠已合并）。每个元素含
  `start_at` / `end_at` 与**两端包含标记** `start_inclusive` / `end_inclusive`：
  起点侧一律包含；被阻塞区间截断的终点包含（该时刻本身可行）；最后一个区间在
  查询终点处不含（半开）。**起止相等且两端都包含**的区间表示一个**孤立可行
  时刻**（两侧恰被阻塞区夹住）。
- `earliest_start_at`：最早可行开始时刻；**无空档时为 `null` 且 `slots` 为空列表**。
- 上下文：`gpus` / `duration_min` / `cluster_gpus` /
  `running_baseline_gpus`（固定基线）/ `query_start_at` / `query_end_at`。

仅 **QUEUED** 作业可查询：RUNNING/COMPLETED/CANCELLED → 409；作业不存在 → 404。

```bash
# 集群 8 GPU：作业 2 GPU × 60 分钟；维护时段 [01:40, 03:20) 预留 7 GPU，
# 则开始时刻落在 (00:40, 03:20) 内都会撞上预留（00:40 起运行恰在 01:40 结束，可行）
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-a","gpus":2,"duration_min":60}'
curl -s -X POST localhost:8000/maintenance-windows -H 'Content-Type: application/json' \
  -d '{"start_at":"2026-09-24T01:40:00Z","end_at":"2026-09-24T03:20:00Z","reserved_gpus":7}'
curl -s -G localhost:8000/jobs/J000001/maintenance-slots \
  --data-urlencode "start=2026-09-24T00:00:00Z" \
  --data-urlencode "end=2026-09-24T05:00:00Z"
```

```jsonc
{
  "job_id": "J000001", "status": "QUEUED", "gpus": 2, "duration_min": 60,
  "cluster_gpus": 8, "running_baseline_gpus": 0,
  "query_start_at": "2026-09-24T00:00:00+00:00",
  "query_end_at": "2026-09-24T05:00:00+00:00",
  "earliest_start_at": "2026-09-24T00:00:00+00:00",
  "slots": [
    {"start_at": "2026-09-24T00:00:00+00:00", "end_at": "2026-09-24T00:40:00+00:00",
     "start_inclusive": true,  "end_inclusive": true},
    {"start_at": "2026-09-24T03:20:00+00:00", "end_at": "2026-09-24T05:00:00+00:00",
     "start_inclusive": true,  "end_inclusive": false}
  ]
}
```

### 端点一览

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/jobs` | 提交作业（自动注册/更新租户权重，支持 `depends_on` 依赖）；可选请求头 **`Idempotency-Key`** 实现按 `tenant_id + key` 的幂等提交（事务内先比对已保存指纹：同参重试 200 返回作业当前对象且无副作用，即使重启后 `CLUSTER_GPUS` 下调使原请求超容量；异参 409，包括新参数本身超容量的情况；仅未命中的首次提交按当前容量校验、失败不占 key），不传则沿用原接口（始终按当前容量校验） |
| POST | `/jobs/graph` | **批量作业图提交（单事务原子落库）**：非空 `jobs` 数组，每项带本批唯一 `ref`、`tenant_id/gpus/duration_min/weight`，`depends_on` 引已有作业 id、`depends_on_refs` 引批内 ref（允许前向及跨租户引用）；同租户多项 weight 不一致报 400；同事务校验资源参数、外部引用与整图环路后分配顺序 id、写租户/作业/依赖；任一错误整批回滚、不改租户权重或队列；返回 `ref_ids`（ref→id）与作业对象，新作业立即遵循现有预演/正式批次依赖判定 |
| GET | `/jobs` | 列作业（可选 `status`、`tenant_id`） |
| GET | `/jobs/{id}` | 作业详情 |
| PATCH | `/jobs/{id}/dependencies` | **整体替换 QUEUED 作业的前置依赖**（去重排序、空数组解除、跨租户允许；未知 id/自依赖/成环 → 400 且原依赖不变，非 QUEUED → 409）；下一次预演/批次按新依赖判断 |
| POST | `/jobs/{id}/cancel` | 取消作业 |
| GET | `/jobs/{id}/cancel-impact` | **取消影响预览（零写入）**：沿跨租户依赖图求传递闭包（穿过已取消节点、不穿过已完成节点），按作业 id 排序返回取消该 QUEUED/RUNNING 作业后会永久受阻的排队后代及各自一条最短依赖链；不存在 404、已完成/已取消 409 |
| POST | `/jobs/{id}/cancel-cascade` | **依赖取消级联清理（落库）**：与取消/依赖修订共用写锁与写事务，事务内重算影响范围，原子取消目标及范围内仍为 QUEUED 的作业并释放目标运行占用；已取消中间节点仅遍历不重复修改；404/409 同上 |
| POST | `/jobs/{id}/complete` | 完成运行中作业并释放 GPU |
| GET | `/jobs/{id}/maintenance-slots` | **QUEUED 作业维护空档查询**：以当前 RUNNING 占用为固定基线，叠加已保存维护时段，返回 `[start, end)` 内可行的预计开始时刻区间（含两端包含标记与孤立时刻）及最早时刻；非 QUEUED → 409，无空档返回空列表 |
| POST | `/tenants` | 创建租户或更新权重；可选 `max_gpus` 设置/更新/清除并发 GPU 上限，可选 `budget_gpu_min` 设置/更新/清除滚动 24h GPU-分钟预算（省略保留、null 清除） |
| GET | `/tenants` / `/tenants/{id}` | 租户与余额 |
| POST | `/scheduler/batch` | **调度下一批（落库）**，返回选择顺序与余额；可选请求体 `{"now": <UTC ISO-8601>}` 指定参考时刻 |
| POST | `/scheduler/preview` | **预演下一批（零写入，事务回滚）**：与 batch 同一套 WDRR/老化/上限/依赖判定，返回预测的派发顺序、批次后 GPU 占用、租户余额与信用事件；可选 `{"now": ...}` |
| GET | `/scheduler/state` | 集群/调参/租户快照 |
| GET | `/queue` | 每租户 FIFO 队列与位置 |
| GET | `/usage/gpu` | **按 UTC 时间窗汇总**每租户/全局 GPU-分钟与重叠作业数（可选 `tenant_id`） |
| POST | `/maintenance-windows` | 创建**持久化 GPU 维护时段**（UTC 半开区间 + 预留 GPU 数）；结束不晚于开始、单条预留超容量、或候选覆盖期间与已有时段的合计预留峰值超当前容量（写事务内校验，端点相接不累加、等于容量可创建）均返回 400 且不落库 |
| GET | `/maintenance-windows` | 列出维护时段（可按 `active_at` UTC 时刻过滤活动时段） |
| GET | `/ledger` | 信用/扣减审计流水 |
| GET | `/health` | 健康检查（本机：<http://localhost:8000/health>） |

---

## 端到端示例

```bash
# alice: 2 GPU × 30 分钟 = 60 成本，权重 1；bob: 6 GPU × 10 = 60 成本，权重 2
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"alice","gpus":2,"duration_min":30,"weight":1}'
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"bob","gpus":6,"duration_min":10,"weight":2}'
# carol 需要 4 GPU，此刻放不下
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"carol","gpus":4,"duration_min":10}'

curl -s -X POST localhost:8000/scheduler/preview  # 先预演：预测派发 [J1,J2]、占满 8 GPU，但零写入
curl -s -X POST localhost:8000/scheduler/batch        # => [J000001, J000002]，用满 8 GPU
curl -s -X POST localhost:8000/scheduler/batch        # => []（容量不足，carol 等待并老化）
curl -s -X POST localhost:8000/jobs/J000001/complete  # 释放 2，仍不足 4
curl -s -X POST localhost:8000/jobs/J000002/complete  # 再释放 6
curl -s -X POST localhost:8000/scheduler/batch        # => [J000003]，carol 获得等待老化补偿
curl -s localhost:8000/queue

# 作业依赖（跨租户）：dave 的作业等 carol 的 J000003 完成后才会被调度
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"dave","gpus":2,"duration_min":15,"depends_on":["J000003"]}'
curl -s -X POST localhost:8000/scheduler/batch        # => []（J000003 仍 RUNNING，dave 不消耗量子）
curl -s localhost:8000/jobs/J000004                   # blocked_by 标明 J000003/RUNNING

# 取消影响预览（零写入）：若取消 J000003，哪些排队后代会永久受阻
curl -s localhost:8000/jobs/J000003/cancel-impact
# {"affected_count":1,"affected_jobs":[{"id":"J000004","chain":["J000003","J000004"],...}],...}

# 依赖修订（仅 QUEUED）：改为等 J000002，或传 [] 直接解除全部依赖；
# 下一次预演/批次即按新依赖判断
curl -s -X PATCH localhost:8000/jobs/J000004/dependencies \
  -H 'Content-Type: application/json' -d '{"depends_on":["J000002"]}'
curl -s -X PATCH localhost:8000/jobs/J000004/dependencies \
  -H 'Content-Type: application/json' -d '{"depends_on":[]}'

curl -s -X POST localhost:8000/jobs/J000003/complete

# 租户并发 GPU 上限：team-c 上限 2，其 2-GPU 作业可派发，后续作业被本租户上限挡住，
# 但其他租户不受影响
curl -s -X POST localhost:8000/tenants -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-c","weight":1,"max_gpus":2}'
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-c","gpus":2,"duration_min":10}'   # 占满上限
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-c","gpus":1,"duration_min":10}'   # 超上限，排队
curl -s -X POST localhost:8000/scheduler/batch             # 只派发第一个
curl -s localhost:8000/tenants/team-c                      # max_gpus=2, running_gpus=2
# 完成后释放额度，下一批次派发第二个；也可再次 POST /tenants 调高或清除上限
curl -s -X POST localhost:8000/scheduler/batch        # => [J000004]

# 租户滚动 24 小时 GPU-分钟预算：team-d 预算 100 GPU-分钟；
# 1 GPU × 60 min 的作业可派发，再来一个同成本作业（累计 120 > 100）排队等待，
# 但其他租户不受影响；完成/取消不退还，24 小时后旧派发滑出窗口额度恢复
curl -s -X POST localhost:8000/tenants -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-d","weight":1,"budget_gpu_min":100}'
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-d","gpus":1,"duration_min":60}'
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"team-d","gpus":1,"duration_min":60}'
curl -s -X POST localhost:8000/scheduler/batch        # => [J....] 只派发第一个
curl -s localhost:8000/tenants/team-d
# {"budget_gpu_min":100,"budget_used_gpu_min":60,"budget_remaining_gpu_min":40,
#  "budget_window_start_at":"...(now-24h)...","budget_window_end_at":"...now(UTC)..."}

# 按 UTC 时间窗汇总实际 GPU 用量（已结束作业取 scheduled_at→completed_at，
# 仍 RUNNING 的作业截断到 end；QUEUED 不计；左闭右开）
curl -s -G localhost:8000/usage/gpu \
  --data-urlencode "start=2026-09-23T00:00:00Z" \
  --data-urlencode "end=2026-09-23T02:00:00Z"
# {"global":{"gpu_minutes":...,"overlapping_jobs":...},"tenants":[...]}

# GPU 维护时段（持久化，本机 http://localhost:8000）：预留 2 GPU，时段内
# RUNNING 不被抢占，新区间重叠的候选作业按容量减去峰值预留量准入
curl -s -X POST localhost:8000/maintenance-windows -H 'Content-Type: application/json' \
  -d '{"start_at":"2026-09-24T02:00:00Z","end_at":"2026-09-24T04:00:00Z","reserved_gpus":2}'
curl -s localhost:8000/maintenance-windows
curl -s -G localhost:8000/maintenance-windows \
  --data-urlencode "active_at=2026-09-24T03:00:00Z"

# 依赖取消级联（执行）：J000010 依赖 J000009；一个调用原子取消目标及其
# 仍为 QUEUED 的全部受阻后代（已取消的中间节点只用于遍历，不被重复修改）
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"eve","gpus":2,"duration_min":30}'
curl -s -X POST localhost:8000/jobs -H 'Content-Type: application/json' \
  -d '{"tenant_id":"frank","gpus":1,"duration_min":10,"depends_on":["J000009"]}'
curl -s -X POST localhost:8000/jobs/J000009/cancel-cascade
# {"cancelled_count":2,"cancelled":["J000009","J000010"],...}
curl -s localhost:8000/jobs/J000010                   # status: CANCELLED

# 批量作业图提交（本机 http://localhost:8000）：单事务原子创建一组带前向与
# 跨租户依赖的作业，返回 ref -> id 映射；任一错误整批回滚、权重与队列不变
curl -s -X POST localhost:8000/jobs/graph \
  -H 'Content-Type: application/json' \
  -d '{"jobs":[
        {"ref":"prep","tenant_id":"gina","gpus":1,"duration_min":10,"weight":1},
        {"ref":"train","tenant_id":"gina","gpus":2,"duration_min":20,"weight":1,
         "depends_on_refs":["prep"]},
        {"ref":"report","tenant_id":"hank","gpus":1,"duration_min":5,"weight":1,
         "depends_on_refs":["train"]}
      ]}'
# {"ref_ids":{"prep":"J000011","train":"J000012","report":"J000013"},
#  "jobs":[ {"id":"J000011",...}, {"id":"J000012","depends_on":["J000011"],...},
#           {"id":"J000013","depends_on":["J000012"],...} ]}
curl -s -X POST localhost:8000/scheduler/preview     # 新作业立即按现有依赖判定
```

---

## 本地开发（不使用 Docker）

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt

# 运行全部测试（220 个：服务层算法/依赖/批量作业图提交/依赖修订/依赖取消级联/用量汇总/租户 GPU 上限/
# 滚动 24h GPU-分钟预算/批次预演/GPU 维护时段（含累计预留峰值校验）/维护空档查询/POST /jobs 幂等提交（含容量下调后重试判定顺序）/依赖就绪老化 + HTTP 端到端）
.venv/bin/python -m pytest -q

# 无需第三方库的核心算法验证脚本（仅用标准库 sqlite3）
python3 scripts/verify_core.py

# 本地启动（默认 SQLite 写入 ./data/scheduler.db）
CLUSTER_GPUS=8 .venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

## 项目结构

```
app/
  config.py       # 环境变量配置
  db.py           # SQLite 连接、schema（tenants/jobs/batches/ledger/job_dependencies/idempotent_jobs/maintenance_windows，含旧库补列迁移）
  scheduler.py    # 公平排队核心（WDRR + 老化 + 维护时段/空档查询 + 租户上限/24h 预算准入 + 批量作业图提交 + 依赖修订 + 依赖取消级联 + 幂等提交，框架无关）
  main.py         # FastAPI 路由与请求模型（POST /jobs 可选 Idempotency-Key 请求头；POST /jobs/graph 批量作业图；PATCH /jobs/{id}/dependencies；GET /jobs/{id}/maintenance-slots；GET /jobs/{id}/cancel-impact 与 POST /jobs/{id}/cancel-cascade）
scripts/
  verify_core.py  # 零依赖核心算法验证
tests/
  test_scheduler.py    # 算法测试（注入时间，确定性断言）
  test_job_graph.py    # 批量作业图提交：顺序 id/ref 映射、前向与跨租户引用、同租户 weight 一致性、整图环路、整批回滚不改权重/队列、预演与批次立即生效
  test_budget.py       # 滚动 24h GPU-分钟预算：设置/保留/清除、超额阻塞、窗口滑动、不退费、预演不落库
  test_preview.py      # 预演测试：与正式批次决策一致、零写入、异常回滚
  test_maintenance.py  # 维护时段：持久化/校验、创建时累计预留峰值校验（重叠累加、相接不累加、等于容量可创建、拒绝不落库）、重叠准入、端点相接、RUNNING 不抢占、预演一致
  test_maintenance_slots.py  # 维护空档查询：可行开始时刻区间合并、端点相接、孤立时刻、固定 RUNNING 基线、HTTP
  test_idempotency.py  # POST /jobs 幂等：首次/重试/409、失败不占 key、依赖规范化、租户隔离、并发与重启只建一个作业、重启后 CLUSTER_GPUS 下调时同参重试仍 200/异参仍 409/新 key 首次才按当前容量校验
  test_dependencies_revision.py  # 依赖修订：整体替换/清空、跨租户、QUEUED 限定、拒绝保持原依赖、预演与批次生效、幂等指纹保留
  test_aging_readiness.py  # 依赖就绪老化：信用只从队首真正就绪时刻累计（最后前置完成/解除阻塞的修订时刻），与老化结算点取较晚值，预演一致且零写入
  test_cancel_cascade.py  # 依赖取消级联：传递闭包、穿过已取消/不穿过已完成、最短依赖链、预览零写入、级联原子取消与 GPU 释放、404/409
  test_api.py          # HTTP 端到端测试
Dockerfile
docker-compose.yml
```
