![robot](img/robot.webp)

# 微调数据构造实验

## 1 xtuner教学使用
详见：https://github.com/InternLM/Tutorial/blob/main/xtuner/README.md
## 2 微调实战

> 采用 **[COIG-CQIA](https://opendatalab.org.cn/OpenDataLab/COIG-CQIA)** **数据集**中的弱智吧数据

### 2.1 概述

#### 2.1.1 **场景需求**

   基于 InternLM-chat-7B 模型，用弱智吧中的数据进行微调，训练一个不弱智的模型

#### 2.1.2 **真实数据预览**

| 问题                                                       | 答案                                                         |
| ---------------------------------------------------------- | ------------------------------------------------------------ |
| 石油也是油，为啥没人用它来炒菜？ | 石油虽然也是一种油，但它并不适用于烹饪。这主要是因为石油和食用油在化学成分和用途上有很大的不同： 1. **化学 成分**：食用油是由植物或动物脂肪制成的，主要含有对人体有益的脂肪酸，如欧米伽脂肪酸。而石油是一种化石燃料，主要由碳氢化合物组成，含有多种有害的化学物质。 2. **毒性**：石油中含有的化学物质对人体是有害的，甚至是致癌的。这些物质在加热时会产 生有毒的烟雾和蒸气，这对健康极为不利。 3. **风味和气味**：食用油在烹饪中可以增添食物的风味，而石油的气味和味道都非常 强烈且令人不悦，不适合用于食物。 4. **燃点**：石油的燃点远低于食用油，这意味着它在较低的温度下就会燃烧，这在厨房中极 为危险，容易引发火灾。 因此，出于健康和安全的考虑，石油绝不能用于烹饪。食用油是专门为食品烹饪设计的，既安全又有益健康。 |
| 据数据表明，异性间的离婚率远大于同性，而中国同性离婚率至今为0这是否说明同性间结婚比异性更有优势|这个问题涉及到对离婚率数据的解读，以及同性和异性婚姻的比较。首先，需要明确一点：中国目前不承认同性婚姻，因此官方统计中不会有关于同性离婚率的数据。因此，任何关于中国同性婚姻离婚率的讨论都是基于非正式或非法律承认的情况。 从更广泛的角度来看，离婚率受到许多因素的影响，包括社会文化、法律环境、经济状况等。比较不同类型婚姻的离婚率时，这些因素都需要被考虑在内。同性婚姻在许多国家和地区是新近承认的，可能受到不同的社会和文化压力，这些都可能影响离婚率。 简单地将两种婚姻形式的离婚率进行对比，并不能直接说明哪一种婚姻形式更有优势。婚姻的质量和稳定性受到众多复杂因素的影响，包括双方的个人特质、沟通能力、外部压力等。因此，解读这样的统计数据时需要谨慎，并考虑到背后的多种可能性和复杂性。 |


### 2.2 数据准备 

#### 2.2.1 数据集下载

可以访问 **[Huggingface](https://huggingface.co/datasets/m-a-p/COIG-CQIA)** 或者 **[OpenDatalab](https://opendatalab.org.cn/OpenDataLab/COIG-CQIA)** 进行数据集下载，由于huggingface网络连接问题，推荐国内的同学访问OpenDataLab进行数据集下载

#### 2.2.2 将数据转为 XTuner 的数据格式
**原始数据格式**
```JSON
{
    "instruction": "天下没有不散的筵席，那么我们相聚的意义又是什么",
    "input": "",
    "output": "尽管我们相聚的时光有限，但是相聚的意义在于创造美好的回忆和珍贵的关系。相聚让我们感受到彼此的关怀、支持和友情。我们可以一起分享喜悦、快乐和困难，互相支持和激励。相聚也可以是一个机会，让我们相互了解、学习和成长。最重要的是，相聚能够带给我们真实的人际交往和情感交流，让我们感受到生活的丰富和美好。所以，即使相聚的时间有限，我们仍然要珍惜这份意义和与彼此相聚的时光。",
    "task_type": {
        "major": [
            "问答"
        ],
        "minor": [
            "逻辑问答",
            "隐喻理解"
        ]
    },
    "domain": [
        "通用"
    ],
    "metadata": "暂无元数据信息",
    "answer_from": "llm",
    "human_verified": true,
    "copyright": "暂无版权及作者信息"
}
```
**目标格式**

```JSON
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

> 采用GPT的编写python转换脚本或采用在线的data analysis功能，即可完成数据的转换，prompt如下：
```markdown
请编写python脚本完成jsonl数据到jsonl数据的格式转换，要求如下：
1）原始jsonl每行格式为：
{
    "instruction": "天下没有不散的筵席，那么我们相聚的意义又是什么",
    "input": "",
    "output": "尽管我们相聚的时光有限，但是相聚的意义在于创造美好的回忆和珍贵的关系。相聚让我们感受到彼此的关怀、支持和友情。我们可以一起分享喜悦、快乐和困难，互相支持和激励。相聚也可以是一个机会，让我们相互了解、学习和成长。最重要的是，相聚能够带给我们真实的人际交往和情感交流，让我们感受到生活的丰富和美好。所以，即使相聚的时间有限，我们仍然要珍惜这份意义和与彼此相聚的时光。",
    "task_type": {
        "major": [
            "问答"
        ],
        "minor": [
            "逻辑问答",
            "隐喻理解"
        ]
    },
    "domain": [
        "通用"
    ],
    "metadata": "暂无元数据信息",
    "answer_from": "llm",
    "human_verified": true,
    "copyright": "暂无版权及作者信息"
}

2）原始jsonl每行格式为：

{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}
3）原始格式的"instruction"对应目标格式的"input"，原始格式的 "output"对应目标格式的 "output"
4）将转换后的数据保存为新的jsonl，中文不要转义
```

> 得到python脚本如下
```python
import json

# 定义原始jsonl文件路径和目标jsonl文件路径
input_file_path = 'original_data.jsonl'
output_file_path = 'transformed_data.jsonl'

# 打开原始jsonl文件进行读取，并打开目标jsonl文件准备写入
with open(input_file_path, 'r', encoding='utf-8') as input_file, \
     open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in input_file:
        # 解析原始数据行为Python字典
        original_data = json.loads(line)
        
        # 构造新的数据格式
        transformed_data = {
            "conversation": [
                {
                    "system": "",
                    "input": original_data.get("instruction", ""),
                    "output": original_data.get("output", "")
                }
            ]
        }
        
        # 将转换后的数据写入新的jsonl文件，确保中文不转义
        json.dump(transformed_data, output_file, ensure_ascii=False)
        output_file.write('\n')  # 在每个json对象后添加换行符，保持jsonl格式

print(f"转换完成，已保存至：{output_file_path}")

```

#### 2.2.3 划分训练集和测试集
>可以直接修改上面的prompt进行格式化并划分，也可以不用划分，修改的prompt如下：

```markdown
请编写python脚本完成jsonl数据到jsonl数据的格式转换，要求如下：
1）原始jsonl每行格式为：
{
    "instruction": "天下没有不散的筵席，那么我们相聚的意义又是什么",
    "input": "",
    "output": "尽管我们相聚的时光有限，但是相聚的意义在于创造美好的回忆和珍贵的关系。相聚让我们感受到彼此的关怀、支持和友情。我们可以一起分享喜悦、快乐和困难，互相支持和激励。相聚也可以是一个机会，让我们相互了解、学习和成长。最重要的是，相聚能够带给我们真实的人际交往和情感交流，让我们感受到生活的丰富和美好。所以，即使相聚的时间有限，我们仍然要珍惜这份意义和与彼此相聚的时光。",
    "task_type": {
        "major": [
            "问答"
        ],
        "minor": [
            "逻辑问答",
            "隐喻理解"
        ]
    },
    "domain": [
        "通用"
    ],
    "metadata": "暂无元数据信息",
    "answer_from": "llm",
    "human_verified": true,
    "copyright": "暂无版权及作者信息"
}

2）原始jsonl每行格式为：

{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}
3）原始格式的"instruction"对应目标格式的"input"，原始格式的 "output"对应目标格式的 "output"
4）将转换后的数据按照7：3的比例划分为训练集和测试集，分别保存为新的jsonl，中文不要转义

```

>得到新的python脚本如下：
```python
import json
from sklearn.model_selection import train_test_split

# 定义原始jsonl文件路径
input_file_path = 'original_data.jsonl'

# 读取并转换数据
transformed_data = []
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    for line in input_file:
        # 将每一行的JSON字符串转换成字典
        original_data = json.loads(line)
        
        # 创建新的数据格式
        new_data = {
            "conversation": [
                {
                    "system": "",
                    "input": original_data.get("instruction", ""),
                    "output": original_data.get("output", "")
                }
            ]
        }
        
        transformed_data.append(new_data)

# 划分数据集
train_data, test_data = train_test_split(transformed_data, test_size=0.3, random_state=42)

# 定义训练集和测试集的文件路径
train_file_path = 'train_data.jsonl'
test_file_path = 'test_data.jsonl'

# 保存训练集
with open(train_file_path, 'w', encoding='utf-8') as train_file:
    for data in train_data:
        json.dump(data, train_file, ensure_ascii=False)
        train_file.write('\n')

# 保存测试集
with open(test_file_path, 'w', encoding='utf-8') as test_file:
    for data in test_data:
        json.dump(data, test_file, ensure_ascii=False)
        test_file.write('\n')

print(f"数据转换完成，训练集已保存至：{train_file_path}，测试集已保存至：{test_file_path}")

```


### 2.3 开始自定义微调
>本节内容可参照 **[xtuner实战](https://github.com/InternLM/Tutorial/blob/main/xtuner/README.md)**，这里简单介绍下流程：

建立文件夹ruozhiba
```bash
mkdir ~/ruozhiba && cd ~/ruozhiba
```

复制internlm-chat-7b模型
```bash
ln -s /share/temp/model_repos/internlm-chat-7b ~/ruozhiba/
```

上传处理后的弱智吧数据

#### 2.3.1 准备配置文件
本案例基于internlm_chat_7b_qlora_oasst1_e3.py进行修改

```bash
# 复制配置文件到当前目录
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .

# 改个文件名
mv internlm_chat_7b_qlora_oasst1_e3_copy.py internlm_chat_7b_qlora_ruozhiba_e3.py

# 修改配置文件内容
vim internlm_chat_7b_qlora_ruozhiba_e3.py
```

减号代表要删除的行，加号代表要增加的行。
```diff
# 修改import部分
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory

# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'

# 修改训练数据为弱智吧训练数据路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = 'train.jsonl'

# 修改 train_dataset 对象
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)
```
#### 2.3.2 **XTuner！启动！**

```bash
xtuner train internlm_chat_7b_ruozhiba.py --deepspeed deepspeed_zero2
```

#### 2.3.3 pth 转 huggingface

将得到的 PTH 模型转换为 HuggingFace 模型，**即：生成 Adapter 文件夹**

`xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH_file_dir} ${SAVE_PATH}`

在本示例中，为：
```bash
mkdir hf
export MKL_SERVICE_FORCE_INTEL=1
export MKL_THREADING_LAYER=GNU
xtuner convert pth_to_hf ./internlm_chat_7b_qlora_ruozhiba_e3.py ./work_dirs/internlm_chat_7b_qlora_ruozhiba_e3/epoch_1.pth ./hf
```

<span style="color: red;">**此时，hf 文件夹即为我们平时所理解的所谓 “LoRA 模型文件”**</span>

> 可以简单理解：LoRA 模型文件 = Adapter

#### 2.4 部署与测试

#### 2.4.1 将 HuggingFace adapter 合并到大语言模型：

```Bash
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```

#### 2.4.2 与合并后的模型对话：
```Bash
# 加载 Adapter 模型对话（Float 16）
xtuner chat ./merged --prompt-template internlm_chat

# 4 bit 量化加载
# xtuner chat ./merged --bits 4 --prompt-template internlm_chat
```

#### 2.4.3 Demo
- 新建cli_demo.py并填入以下内容

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "merged" #模型名称或路径

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("User  >>> ")
    input_text.replace(' ', '')
    if input_text == "exit":
        break
    response, history = model.chat(tokenizer, input_text, history=messages)
    messages.append((input_text, response))
    print(f"robot >>> {response}")
```

- 运行 `cli_demo.py` 以目测微调效果
```bash
python ./cli_demo.py
```

**效果：**

| 微调前 | 微调后 |
| --- | --- |
| ![before](img/before.png) | ![after](imgs/after.png) |

**`xtuner chat`** **的启动参数**

| 启动参数              | 干哈滴                                                       |
| --------------------- | ------------------------------------------------------------ |
| **--prompt-template** | 指定对话模板                                                 |
| --system              | 指定SYSTEM文本                                               |
| --system-template     | 指定SYSTEM模板                                               |
| -**-bits**            | LLM位数                                                      |
| --bot-name            | bot名称                                                      |
| --with-plugins        | 指定要使用的插件                                             |
| **--no-streamer**     | 是否启用流式传输                                             |
| **--lagent**          | 是否使用lagent                                               |
| --command-stop-word   | 命令停止词                                                   |
| --answer-stop-word    | 回答停止词                                                   |
| --offload-folder      | 存放模型权重的文件夹（或者已经卸载模型权重的文件夹）         |
| --max-new-tokens      | 生成文本中允许的最大 `token` 数量                                |
| **--temperature**     | 温度值                                                       |
| --top-k               | 保留用于顶k筛选的最高概率词汇标记数                          |
| --top-p               | 如果设置为小于1的浮点数，仅保留概率相加高于 `top_p` 的最小一组最有可能的标记 |
| --seed                | 用于可重现文本生成的随机种子                                 |



## 3 其他已知问题和解决方案

## 4 注意事项

## 5 作业
1）选一个任务场景：角色扮演、对话助手……
2）收集数据：公开数据集、贴吧论坛、问答网站……
3）数据处理：预处理、格式转换、人工编写回复……
4）使用Xtuner开始微调！


