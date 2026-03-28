import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# 数据预处理函数
def load_and_preprocess_data(csv_path):
    """加载和预处理数据"""
    print("正在加载数据...")
    df = pd.read_csv(csv_path)[:1000]
    
    # 过滤有效数据
    df = df[df['Lsa_summary'].notna() & df['sentiment_deepseek'].notna()]
    df = df[df['sentiment_deepseek'] != 0]  # 移除无效的情感标签
    
    print(f"有效数据数量: {len(df)}")
    print(f"情感分布: {df['sentiment_deepseek'].value_counts().sort_index()}")
    
    return df

def create_prompt_template(text, sentiment, stock_symbol="STOCK"):
    """创建训练提示模板"""
    # 使用与sentiment_deepseek_deepinfra.py相同的对话格式
    system_prompt = "Forget all your previous instructions. You are a financial expert with stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive. 1 summarized news will be passed in each time, you will give score in format as shown below in the response from assistant."
    
    # 构建用户输入
    user_content = f"News to Stock Symbol -- {stock_symbol}: {text}"
    
    # 构建完整的对话
    conversation = f"""System: {system_prompt}

User: News to Stock Symbol -- AAPL: Apple (AAPL) increase 22%
Assistant: 5

User: News to Stock Symbol -- AAPL: Apple (AAPL) price decreased 30%
Assistant: 1

User: News to Stock Symbol -- AAPL: Apple (AAPL) announced iPhone 15
Assistant: 4

User: {user_content}
Assistant: {sentiment}"""
    
    return conversation

def prepare_dataset(df, tokenizer, max_length=512):
    """准备训练数据集"""
    print("正在准备数据集...")
    
    texts = []
    labels = []
    
    for _, row in df.iterrows():
        text = row['Lsa_summary']
        sentiment = int(row['sentiment_deepseek'])
        stock_symbol = row.get('Stock_symbol', 'STOCK')  # 获取股票符号，如果没有则使用默认值
        
        if pd.isna(text) or text == '':
            continue
            
        prompt = create_prompt_template(text, sentiment, stock_symbol)
        texts.append(prompt)
        labels.append(sentiment)
    
    # 分割训练集和验证集 (80% 训练, 20% 验证)
    train_texts, eval_texts, train_labels, eval_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=None
    )
    
    print(f"训练集大小: {len(train_texts)}")
    print(f"验证集大小: {len(eval_texts)}")
    
    # 创建训练数据集
    train_dataset = Dataset.from_dict({
        'text': train_texts,
        'label': train_labels
    })
    
    # 创建验证数据集
    eval_dataset = Dataset.from_dict({
        'text': eval_texts,
        'label': eval_labels
    })
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

        labels = tokenized['input_ids'].clone()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

        for i, text in enumerate(examples['text']):
            label_value = str(examples['label'][i])

            label_ids = tokenizer.encode(label_value, add_special_tokens=False)
            input_ids = labels[i]
            actual_length = (input_ids != pad_token_id).sum().item()

            # 先全部屏蔽
            labels[i, :] = -100

            # 只保留最后的 label token 区间
            label_len = len(label_ids)
            if label_len <= actual_length:
                start = actual_length - label_len
                end = actual_length
                labels[i, start:end] = input_ids[start:end]

        tokenized['labels'] = labels
        return tokenized

    
    # 对训练集和验证集进行tokenization
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    eval_tokenized = eval_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    return train_tokenized, eval_tokenized
def create_tokenizer_only():
    """只加载 tokenizer，用于验证数据和 tokenization 流程"""
    print("正在加载 tokenizer...")
    
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return tokenizer


def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir="./qwen_sentiment_model"):
    """训练模型"""
    print("开始训练模型...")
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        # report_to=None,  # 禁用wandb等报告工具
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    print(f"模型已保存到: {output_dir}")
def main():
    """只验证数据准备与 tokenization，不训练模型"""
    csv_path = "nasdaq_news_sentiment/1.csv"
    
    df = load_and_preprocess_data(csv_path)
    tokenizer = create_tokenizer_only()
    train_dataset, eval_dataset = prepare_dataset(df, tokenizer)

    print("数据准备完成")
    print("训练集大小:", len(train_dataset))
    print("验证集大小:", len(eval_dataset))

    sample = train_dataset[0]
    print("sample keys:", sample.keys())

    input_ids = sample["input_ids"]
    labels = sample["labels"]

    effective_ids = [x for x in labels if x != -100]
    print("有效 label token 数:", len(effective_ids))
    print("有效 label token 解码:", repr(tokenizer.decode(effective_ids)))


if __name__ == "__main__":
    main() 
