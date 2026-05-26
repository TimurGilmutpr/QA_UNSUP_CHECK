import torch
import json
import warnings
import joblib
warnings.filterwarnings("ignore", category=DeprecationWarning)
from datasets import load_dataset, Dataset
from tqdm import tqdm
import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

from RUN_AND_LEARN_LIB import load_model  # finetune_lora больше не используется
from RAG_LIB import build_rag_index, rag_generate
from METRICS_LIB import (
    compute_perplexity,
    compute_bleu,
    compute_tok_per_word,
    compute_latency,
    compute_token_speed
)
from EXCEL_SAMPLE import write_metrics_to_excel
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

import gc
import torch
import shutil
import os
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import faiss
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
import math
import pynvml
from peft import PeftModel
# ========================== ПАРАМЕТРЫ ==========================
EXCEL_DATA_FOLDER = "ПРИМЕРЫ"               # папка с Excel файлами (вопрос/ответ)
VALIDATION_SPLIT = 0.1                       # доля валидационной выборки
SYSTEM_PROMPT = """Вы — эксперт-цитогенетик. ВАШ ОТВЕТ ДОЛЖЕН БЫТЬ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ.
Запрещено использовать украинский, белорусский или любой другой язык.
Если вы ответите не на русском, это будет считаться ошибкой."""


SYSTEM_PROMPT = """Вы — эксперт-цитогенетик. Необходимо ответить на русском языке, дать полную консультацию пациенту о его кариотипе и его особенностях"""


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_LIST = [
    "meta-llama/Llama-3.1-8B-Instruct"
]

TRAIN_PATH = "JSON_MARKED/train.jsonl"
VAL_PATH = "JSON_MARKED/val.jsonl"

EXCEL_PATH = "Таблица метрик.xlsx"


# ----------------------------------------------------------------------
# Функция загрузки данных из Excel

def load_data_from_excel_folder(folder: str, validation_split: float = 0.1):
    """
    Загружает все Excel файлы из папки, ожидая колонки 'Вопрос' и 'Ответ'.
    Возвращает:
        train_dataset (Dataset)   : для SFT (с formatted text)
        val_dataset (Dataset)     : для SFT
        corpus (list[str])        : все ответы из train (для RAG)
        val_prompts (list[str])   : вопросы валидации (для оценки)
        val_references (list[str]): ответы валидации (для оценки)
    """
    all_questions = []
    all_answers = []
    excel_list = [f"ПРИМЕРЫ/{k}" for k in os.listdir("ПРИМЕРЫ") if k.endswith("xlsx") and "~" not in k]
    dfs_all = []
    for k in excel_list:
        df = pd.read_excel(k)
        dfs_all.append(df)
    df_all = pd.concat(dfs_all, ignore_index=True)


    X = df_all[['Вопрос']]   # DataFrame
    y = df_all['Ответ']      # Series


    train_q, val_q, train_a, val_a = train_test_split(
        X, y,
        test_size=validation_split,
        random_state=42,
        shuffle=True,
    )

    # Извлекаем значения как обычные списки строк
    train_q = train_q.iloc[:, 0].tolist()   # или train_q.squeeze().tolist()
    val_q   = val_q.iloc[:, 0].tolist()
    # train_a и val_a уже Series – их можно преобразовать в список:
    train_a = train_a.tolist()
    val_a   = val_a.tolist()

    # Форматирование для обучения (с ChatML разметкой Llama-3.1)
    def format_chat(question, answer):
        return {
            "text": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>"
        }

    train_dataset = Dataset.from_list([format_chat(q, a) for q, a in zip(train_q, train_a)])
    val_dataset = Dataset.from_list([format_chat(q, a) for q, a in zip(val_q, val_a)])

    # Корпус для RAG (берём все ответы из обучающей выборки)
    corpus = train_a

    return train_dataset, val_dataset, corpus, val_q, val_a


# ----------------------------------------------------------------------
# Вспомогательная функция для форматирования промпта при генерации
# ----------------------------------------------------------------------
def format_prompt_for_generation(user_question: str) -> str:
    """Формирует промпт для генерации без ответа ассистента."""
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"






class CompletionOnlyLMDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, response_template, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response_template = response_template
        self.response_token_ids = self.tokenizer.encode(self.response_template, add_special_tokens=False)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"]
        input_ids = batch["input_ids"]
        for i in range(len(input_ids)):
            seq = input_ids[i].tolist()
            start_idx = None
            for j in range(len(seq) - len(self.response_token_ids) + 1):
                if seq[j:j+len(self.response_token_ids)] == self.response_token_ids:
                    start_idx = j
                    break
            if start_idx is not None:
                labels[i, :start_idx + len(self.response_token_ids)] = -100
            else:
                labels[i, :] = -100  # если шаблон не найден (аварийно)
        return batch

def save_test_promts_excel_to_calculate_adequacy_of_answers(promts: list,references: list, answers: list, f_name: str) -> pd.DataFrame:
    
    df = pd.DataFrame({
        "Promts" : promts,
        "References" : references,
        "Answers" : answers
    })
    
    df.to_excel(f_name)





def get_gpu_name(device_id: int = None) -> str:
    """
    Возвращает название GPU, доступной в системе через PyTorch.

    Параметры:
        device_id (int, optional): Индекс GPU (0, 1, ...). Если не указан,
            используется текущее активное устройство (torch.cuda.current_device()).

    Возвращает:
        str: Название GPU или "CPU", если CUDA недоступна.

    Исключения:
        RuntimeError: Если указан неверный device_id или CUDA недоступна при запросе конкретного устройства.
    """
    if not torch.cuda.is_available():
        return "CPU"

    if device_id is None:
        device_id = torch.cuda.current_device()
    else:
        # Проверяем, существует ли устройство с таким индексом
        if device_id < 0 or device_id >= torch.cuda.device_count():
            raise RuntimeError(f"Устройства с индексом {device_id} не существует. Доступно устройств: {torch.cuda.device_count()}")

    return torch.cuda.get_device_name(device_id)

def cleanup_cuda():
    """Очистка Python GC и CUDA кэшей. ВАЖНО: внешние ссылки на model/tokenizer надо удалить в месте вызова."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
def mem(tag=""):
    """Печатает текущее использование VRAM (allocated/reserved) и пик."""
    if not torch.cuda.is_available():
        print(f"[{tag}] CUDA not available")
        return
    torch.cuda.synchronize()
    a = torch.cuda.memory_allocated()/1024**2
    r = torch.cuda.memory_reserved()/1024**2
    mx = torch.cuda.max_memory_allocated()/1024**2
    print(f"[{tag}] allocated={a:.1f}MB reserved={r:.1f}MB max_alloc={mx:.1f}MB")


def load_validation_texts():
    dataset = load_dataset("json", data_files={"validation": VAL_PATH})
    texts = []
    references = []

    for ex in dataset["validation"]:
        user_msg = ex["messages"][1]["content"]
        assistant_msg = ex["messages"][2]["content"]

        texts.append(user_msg)
        references.append(assistant_msg)

    return texts, references

def get_power_usage(handle):
    """Возвращает текущее энергопотребление GPU в ваттах (float)"""
    # Функция возвращает значение в милливаттах, переводим в ватты
    return pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0


def run_clear(model_name, token = "", quantization_config=None, max_new_tokens = 1300, test_status = False):
    print(f"\n===== CLEAR: {model_name} =====")

    model, tokenizer = load_model(
        model_name=model_name,
        device=DEVICE,
        quantization=quantization_config,
        token = token
    )

    prompts, references = load_validation_texts()
    predictions = []

    if test_status:
        prompts = [prompts[0]]
        
    # Инициализация NVML (для доступа к энергопотреблению GPU)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # предположим, используется GPU 0, т.к. у нас только 1 gpu


    losses = []
    times = []
    generated_lens = []
    predictions = []
    energy_consumed = []  # список накопленной энергии в джоулях для каждого промпта

    model.eval()

    for prompt in tqdm(prompts, desc="Обработка промптов"):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 1. Loss на промпте (для перплексии)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())

        # 2. Замер энергии ДО генерации
        power_start = get_power_usage(handle)
        time_start = time.time()

        # 3. Генерация ответа
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )

        # 4. Замер энергии ПОСЛЕ генерации
        time_end = time.time()
        power_end = get_power_usage(handle)

        elapsed = time_end - time_start
        times.append(elapsed)

        # 5. Количество сгенерированных токенов
        generated_len = output_ids.shape[1] - inputs['input_ids'].shape[1]
        generated_lens.append(generated_len)

        # 6. Расчёт потреблённой энергии (интеграл мощности по времени, аппроксимация средним)
        avg_power = (power_start + power_end) / 2.0
        energy_joules = avg_power * elapsed  # E = P * t (Вт * с = Дж)
        energy_consumed.append(energy_joules)

        # 7. Декодированный ответ
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(text)

    # Сохранение в Excel
    quant_str = ""
    if quantization_config:
        quant_str = quantization_config
    f_name = f"./TEST_ANSWERS/clear_{model_name.replace('/', '--')}_{quant_str}_tokens_{max_new_tokens}.xlsx"
    save_test_promts_excel_to_calculate_adequacy_of_answers(promts=prompts, references=references,answers=predictions, f_name=f_name)

    # Вычисление метрик
    perplexity = math.exp(np.mean(losses))
    blue = compute_bleu(references, predictions)  # предполагается, что references доступны
    latency = float(np.mean(times))
    token_speed = float(np.mean([l / t for l, t in zip(generated_lens, times) if t > 0]))

    # Метрика энергоэффективности: токенов на джоуль (или на ватт-час)
    # Если потреблённая энергия очень мала, можно избежать деления на ноль
    epsilon = 1e-6
    tokens_per_joule = float(np.mean([g / (e + epsilon) for g, e in zip(generated_lens, energy_consumed)]))
    # Для перевода в токены на ватт-час: 1 ватт-час = 3600 джоулей
    tokens_per_wh = tokens_per_joule * 3600


    metrics = {
        "perplexity": perplexity,
        "blue": blue,
        "lat": latency,
        "tok/s": token_speed,
        "tok/j": tokens_per_joule,
    }

    write_metrics_to_excel(EXCEL_PATH, model_name, "clear", metrics, gpu_name=get_gpu_name(), quant=quantization_config, test_path=f_name)
    mem("after CLEAR")

    del model
    del tokenizer
    cleanup_cuda()
    mem("after CLEAR cleanup")
    return metrics


# ----------------------------------------------------------------------
# ФУНКЦИЯ run_rag (использует корпус из Excel)
# ----------------------------------------------------------------------
def run_rag(model_name,
            token="",
            quantization_config=None,
            index_path="faiss_index.bin",
            max_new_tokens=1300,
            test_status=False):
    print(f"\n===== RAG: {model_name} =====")

    # Загружаем данные из Excel, чтобы получить корпус и валидацию
    train_dataset, val_dataset, corpus, val_questions, val_references = load_data_from_excel_folder(
        EXCEL_DATA_FOLDER, validation_split=VALIDATION_SPLIT
    )
    if test_status:
        val_questions = [val_questions[0]]
        val_references = [val_references[0]]

    # Загрузка модели
    model, tokenizer = load_model(
        model_name=model_name,
        device=DEVICE,
        quantization=quantization_config,
        token=token
    )

    # Построение или загрузка индекса FAISS
    if os.path.exists(index_path):
        print(f"Загрузка готового индекса из {index_path}")
        index = faiss.read_index(index_path)
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        print("Построение нового индекса...")
        index, embedder = build_rag_index(corpus)
        faiss.write_index(index, index_path)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    losses = []
    times = []
    generated_lens = []
    predictions = []
    energy_consumed = []

    model.eval()

    for prompt in tqdm(val_questions, desc="Обработка промптов RAG"):
        # Поиск контекста
        query_vec = embedder.encode([prompt])
        distances, indices = index.search(query_vec, 3)
        context = "\n".join([corpus[i] for i in indices[0]])

        # Формируем RAG-промпт (без system, просто контекст + вопрос)
        rag_prompt = f"Контекст:\n{context}\n\nВопрос:\n{prompt}\nОтвет:"
        inputs = tokenizer(rag_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())

        power_start = get_power_usage(handle)
        time_start = time.time()

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True
        )

        time_end = time.time()
        power_end = get_power_usage(handle)
        elapsed = time_end - time_start
        times.append(elapsed)

        generated_len = output_ids.shape[1] - inputs['input_ids'].shape[1]
        generated_lens.append(generated_len)

        avg_power = (power_start + power_end) / 2.0
        energy_joules = avg_power * elapsed
        energy_consumed.append(energy_joules)

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if full_text.startswith(rag_prompt):
            answer = full_text[len(rag_prompt):].lstrip()
        else:
            answer = full_text
        predictions.append(answer)

    quant_str = quantization_config or ""
    f_name = f"./TEST_ANSWERS/RAG_{model_name.replace('/', '--')}_{quant_str}_tokens_{max_new_tokens}.xlsx"
    save_test_promts_excel_to_calculate_adequacy_of_answers(
        promts=val_questions,
        references=val_references,
        answers=predictions,
        f_name=f_name
    )

    perplexity = math.exp(np.mean(losses))
    blue = compute_bleu(val_references, predictions)
    latency = float(np.mean(times))
    token_speed = float(np.mean([l / t for l, t in zip(generated_lens, times) if t > 0]))
    epsilon = 1e-6
    tokens_per_joule = float(np.mean([g / (e + epsilon) for g, e in zip(generated_lens, energy_consumed)]))

    metrics = {
        "perplexity": perplexity,
        "blue": blue,
        "lat": latency,
        "tok/s": token_speed,
        "tok/j": tokens_per_joule,
    }

    write_metrics_to_excel(EXCEL_PATH, model_name, "RAG", metrics,
                           gpu_name=get_gpu_name(), quant=quantization_config, test_path=f_name)

    mem("after RAG")
    del model, tokenizer, index, embedder
    cleanup_cuda()
    mem("after RAG cleanup")

    return metrics

# ----------------------------------------------------------------------
# ФУНКЦИЯ run_lora (использует данные из Excel и SFTTrainer)
# ----------------------------------------------------------------------


def cleanup_cuda():
    """Очистка Python GC и CUDA кэшей. Используйте после del model/tokenizer/..."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()



def run_lora(model_name,
             token="",
             quantization_config=None,
             max_new_tokens=1300,
             num_epoch=10,
             batch_size = 4,
             grad_accum_steps = 4,
             lr = 2e-4,
             test_status=False,
             optim="adamw_torch_fused",
             test_trained = None,
             lora_r=16,
             lora_alpha=32,
             lora_dropout=0.05,
             target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens"],#["q_proj", "v_proj", "k_proj", "o_proj"]


             ):
    print(f"\n===== LORA: {model_name} =====")

    quant = None
    if quantization_config:
        q_dict = {
            "4bit": BitsAndBytesConfig(load_in_4bit=True),
            "8bit": BitsAndBytesConfig(load_in_8bit=True)
        }
        quant = q_dict[quantization_config]
    # 1. Загрузка данных из Excel
    train_dataset, val_dataset, corpus, val_questions, val_references = load_data_from_excel_folder(
        EXCEL_DATA_FOLDER, validation_split=VALIDATION_SPLIT
    )
    if not test_trained:
        # 2. Загрузка модели и токенизатора


        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            dtype=torch.bfloat16,
            quantization_config=quant,
            token=token
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        # 3. Конфигурация LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )


        # 4. Аргументы обучения (SFTConfig)
        modules_list = "_".join(target_modules)
        output_dir = f"{model_name.replace('/', '_')}_lora_epochs_{num_epoch}_batch_{batch_size}_optim_{optim}_r_{lora_r}_alpha_{lora_alpha}_lr_{lr}_drop_out_{lora_dropout}_{modules_list}"
        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            num_train_epochs=num_epoch,
            lr_scheduler_type="cosine",
            learning_rate=lr,
            gradient_checkpointing=True,
            weight_decay=0.01,
            max_grad_norm=1.0,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            report_to="none",
            optim=optim,
            #response_template="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

        #Создание SFTTrainer
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=lora_config,
        )
        
        # Запуск обучения
        trainer.train()

        # Сохранение адаптеров
        lora_dir = output_dir
        test_trained = lora_dir
        trainer.model.save_pretrained(lora_dir)
        tokenizer.save_pretrained(lora_dir)

        model.eval()
        cleanup_cuda()
        del model
        del tokenizer
        del trainer
    
    

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        token=token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    if test_trained:
        model = PeftModel.from_pretrained(base_model, test_trained)
        model.eval()
    else:
        model = trainer.model
        model.eval()

    #
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    losses = []
    times = []
    generated_lens = []
    predictions = []
    energy_consumed = []


    if test_status:
        val_questions, val_references = [val_questions[0]], [val_references[0]]
    print("")
    count = 1
    for prompt in tqdm(val_questions, desc="Оценка LoRA"):
        # Форматируем промпт для генерации
        formatted_prompt = format_prompt_for_generation(prompt)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

        # Loss на промпте (перплексия)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())

        # Замер энергии
        power_start = get_power_usage(handle)
        time_start = time.time()

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
            #eos_token_id=tokenizer.eos_token_id,
            #temperature=0.3, 
            #early_stopping=True
        )

        time_end = time.time()
        power_end = get_power_usage(handle)
        elapsed = time_end - time_start
        times.append(elapsed)

        generated_len = output_ids.shape[1] - inputs['input_ids'].shape[1]
        generated_lens.append(generated_len)

        avg_power = (power_start + power_end) / 2.0
        energy_joules = avg_power * elapsed
        energy_consumed.append(energy_joules)

        # Декодируем только ответ (отсекаем промпт)
        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Ищем начало ответа ассистента
        assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if assistant_marker in full_text:
            answer = full_text.split(assistant_marker)[-1].strip()
        else:
            answer = full_text
        predictions.append(answer)
        count += 1
        #if count == 1 or count%5==0:
        #    print("Log answer in test")
        #    print(answer)

    # Сохранение ответов в Excel
    quant_str = quantization_config or ""
    f_name = f"./TEST_ANSWERS/LORA_{model_name.replace('/', '--')}_{quant_str}_tokens_{max_new_tokens}_epochs_{num_epoch}.xlsx"
    if test_trained:
        f_name = f"./TEST_ANSWERS/LORA_{test_trained}.xlsx"
    save_test_promts_excel_to_calculate_adequacy_of_answers(
        promts=val_questions,
        references=val_references,
        answers=predictions,
        f_name=f_name
    )

    # Расчёт метрик
    perplexity = math.exp(np.mean(losses))
    blue = compute_bleu(val_references, predictions)
    latency = float(np.mean(times))
    token_speed = float(np.mean([l / t for l, t in zip(generated_lens, times) if t > 0]))
    epsilon = 1e-6
    tokens_per_joule = float(np.mean([g / (e + epsilon) for g, e in zip(generated_lens, energy_consumed)]))

    metrics = {
        "perplexity": perplexity,
        "blue": blue,
        "lat": latency,
        "tok/s": token_speed,
        "tok/j": tokens_per_joule,
    }

    write_metrics_to_excel(EXCEL_PATH, model_name, "LORA", metrics,
                           gpu_name=get_gpu_name(), path=test_trained,
                           quant=quantization_config, test_path=f_name)

    mem("after LoRA")
    del model, tokenizer
    cleanup_cuda()
    mem("after LoRA cleanup")

    return metrics







def run_full_finetune(
    model_name,
    token="",
    max_new_tokens=1300,
    num_epoch=5,
    test_status=False,
    learn_len=1536,
    batch_size=3,
    device_batch_size=3,
    padding=False,
    data_collator_type="new",
    test_trained = None,
    name = None,
    make_test = False
):
    print(f"\n===== FULL FINETUNE: {model_name} =====")

    # Загрузка данных из Excel (возвращает уже отформатированные Dataset)
    train_dataset, val_dataset, corpus, val_questions, val_references = load_data_from_excel_folder(
        EXCEL_DATA_FOLDER, validation_split=VALIDATION_SPLIT
    )
    if test_status:
        val_questions = [val_questions[0]]
        val_references = [val_references[0]]
    final_dir = None
    # Загрузка модели (без квантизации)
    if not test_trained:
        model, tokenizer = load_model(
            model_name=model_name,
            device="cuda",
            quantization=None,
            torch_dtype=torch.bfloat16,
            token=token
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        # Токенизация датасетов (они уже в текстовом виде, нужно преобразовать в input_ids)
        # Но для full finetune мы можем использовать стандартный Trainer и DataCollator,
        # поэтому преобразуем тексты в токены с созданием поля labels.
        def tokenize_fn(examples):
            # Токенизируем тексты (они уже содержат полную разметку)
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                max_length=learn_len,
                padding=padding
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        train_tok = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
        val_tok = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

        # Директория для сохранения
        out_dir = f"{model_name.replace('/', '_')}_full_finetune_epochs_{num_epoch}_maxlen_{learn_len}_bs_{batch_size}"

        # Data collator
        if data_collator_type == "new":
            response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            data_collator = CompletionOnlyLMDataCollator(
                response_template=response_template,
                tokenizer=tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
        else:
            data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

        args = TrainingArguments(
            output_dir=out_dir,
            bf16=True,
            per_device_train_batch_size=device_batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=16,
            learning_rate=1e-5,
            warmup_steps=0.03,
            weight_decay=0.1,
            num_train_epochs=num_epoch,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="adamw_torch_fused",
            report_to="none"
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_tok,
            eval_dataset=val_tok,
            data_collator=data_collator,
        )

        trainer.train()
        collator_str = "old"
        if data_collator_type=="new":
            collator_str = "new"
        final_dir = f"{out_dir}_collator_{collator_str}_save"
        trainer.save_model(final_dir)
        tokenizer.save_pretrained(final_dir)

        # Очистка и перезагрузка для оценки
        #mem("after Full Finetune")
        del model, tokenizer
        cleanup_cuda()
        #mem("after Full Finetune cleanup")

    else:
        final_dir = test_trained
    if make_test:
        model = AutoModelForCausalLM.from_pretrained(final_dir, dtype = torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.to(DEVICE)
    
    
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    
        losses = []
        times = []
        generated_lens = []
        predictions = []
        energy_consumed = []
    
        model.eval()
    
        for prompt in tqdm(val_questions, desc="Оценка Full Finetune"):
            formatted_prompt = format_prompt_for_generation(prompt)
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())
    
            power_start = get_power_usage(handle)
            time_start = time.time()
    
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True
            )
    
            time_end = time.time()
            power_end = get_power_usage(handle)
            elapsed = time_end - time_start
            times.append(elapsed)
    
            generated_len = output_ids.shape[1] - inputs['input_ids'].shape[1]
            generated_lens.append(generated_len)
    
            avg_power = (power_start + power_end) / 2.0
            energy_joules = avg_power * elapsed
            energy_consumed.append(energy_joules)
    
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            assistant_marker = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            if assistant_marker in full_text:
                answer = full_text.split(assistant_marker)[-1].strip()
            else:
                answer = full_text
            predictions.append(answer)
    
        f_name = f"./TEST_ANSWERS/Full_finetune_{model_name.replace('/', '--')}_epochs_{num_epoch}_tokens_{max_new_tokens}.xlsx"
        if test_trained:
            test_trained = name
            f_name = f"./TEST_ANSWERS/Full_finetune_{test_trained}.xlsx"
        
        save_test_promts_excel_to_calculate_adequacy_of_answers(
            promts=val_questions,
            references=val_references,
            answers=predictions,
            f_name=f_name
        )
    
        perplexity = math.exp(np.mean(losses))
        blue = compute_bleu(val_references, predictions)
        latency = float(np.mean(times))
        token_speed = float(np.mean([l / t for l, t in zip(generated_lens, times) if t > 0]))
        epsilon = 1e-6
        tokens_per_joule = float(np.mean([g / (e + epsilon) for g, e in zip(generated_lens, energy_consumed)]))
    
        metrics = {
            "perplexity": perplexity,
            "blue": blue,
            "lat": latency,
            "tok/s": token_speed,
            "tok/j": tokens_per_joule,
        }
    
        write_metrics_to_excel(EXCEL_PATH, model_name, "FULL FINETUNE", metrics,
                               gpu_name=get_gpu_name(), path=final_dir, quant=None, test_path=f_name)
    
        mem("after Full Finetune eval")
        del model, tokenizer
        cleanup_cuda()
        mem("after Full Finetune cleanup")
    
        return metrics

# ==================== UNSUPERVISED LoRA (Language Modeling) ====================

def load_unsupervised_corpus(txt_path: str, tokenizer, chunk_size: int = 512, overlap: int = 128, val_split: float = 0.05):
    """
    Загружает текстовый корпус из .txt, разбивает на перекрывающиеся чанки.
    Возвращает train_dataset, val_dataset (HuggingFace Dataset с полем 'text').
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        full_text = f.read()
    
    # Разбиваем на слова/токены для приблизительной нарезки по символам (можно и по токенам позже)
    # Сначала нарежем на чанки по символам с перекрытием
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(full_text), step):
        chunk = full_text[i:i+chunk_size]
        if len(chunk) < 100:   # отбрасываем слишком короткие
            continue
        chunks.append(chunk)
    
    # Токенизируем, чтобы проверить длину в токенах (опционально)
    # Оставим как есть — позже при токенизации датасета обрежем до max_length
    
    from sklearn.model_selection import train_test_split
    train_chunks, val_chunks = train_test_split(chunks, test_size=val_split, random_state=42)
    
    train_dataset = Dataset.from_list([{"text": t} for t in train_chunks])
    val_dataset = Dataset.from_list([{"text": t} for t in val_chunks])
    return train_dataset, val_dataset


def run_unsupervised_lora(
    model_name: str,
    corpus_path: str = "dataset.txt",   # путь к твоему txt-корпусу
    token: str = "",
    quantization_config: str = None,    # "4bit" или "8bit"
    num_epoch: int = 3,
    batch_size: int = 2,
    grad_accum_steps: int = 4,
    lr: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: list = ["q_proj", "v_proj", "k_proj", "o_proj"],
    max_seq_length: int = 1024,
    chunk_size_sym: int = 2000,        # размер чанка в символах (приблизительно)
    overlap_sym: int = 200,
    val_split: float = 0.05,
    optim: str = "adamw_torch_fused",
    test_status: bool = False,          # если True — только 1 пример для быстрой проверки
):
    print(f"\n===== UNSUPERVISED LoRA (LM) on {corpus_path} =====")
    
    # 1. Квантизация
    quant = None
    if quantization_config:
        q_dict = {
            "4bit": BitsAndBytesConfig(load_in_4bit=True),
            "8bit": BitsAndBytesConfig(load_in_8bit=True)
        }
        quant = q_dict.get(quantization_config)
    
    # 2. Загрузка базовой модели и токенизатора
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quant,
        token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = max_seq_length
    
    # 3. Загрузка корпуса
    train_dataset, val_dataset = load_unsupervised_corpus(
        corpus_path, tokenizer,
        chunk_size=chunk_size_sym,
        overlap=overlap_sym,
        val_split=val_split
    )
    
    if test_status:
        train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(5, len(val_dataset))))
    
    # 4. Токенизация датасета
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_overflowing_tokens=False
        )
        
    train_tok = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    val_tok = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    
    # 5. LoRA конфиг
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 6. Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 7. Training arguments
    modules_str = "_".join(target_modules[:3])  # для краткости
    output_dir = f"unsup_{model_name.replace('/', '_')}_epochs_{num_epoch}_r_{lora_r}_alpha_{lora_alpha}_lr_{lr}"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        num_train_epochs=num_epoch,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        optim=optim,
        bf16=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )
    
    # 8. Обучение
    trainer.train()
    
    # 9. Сохранение адаптеров
    lora_dir = output_dir + "_final"
    model.save_pretrained(lora_dir)
    tokenizer.save_pretrained(lora_dir)
    
    # 10. Оценка перплексии на валидации
    eval_loss = trainer.evaluate()["eval_loss"]
    perplexity = math.exp(eval_loss)
    print(f"Validation Perplexity: {perplexity:.2f}")
    
    # 11. Сохраняем метрики в Excel (опционально, используя существующую функцию)
    try:
        metrics = {
            "perplexity": perplexity,
            "blue": None,
            "lat": None,
            "tok/s": None,
            "tok/j": None,
        }
        write_metrics_to_excel(
            EXCEL_PATH, model_name, "UNSUPERVISED_LORA", metrics,
            gpu_name=get_gpu_name(), path=lora_dir, quant=quantization_config, test_path=corpus_path
        )
    except:
        print("Не удалось записать метрики в Excel (функция может требовать дополнительных параметров)")
    
    # 12. Очистка
    del model, tokenizer, trainer
    cleanup_cuda()
    mem("after Unsupervised LoRA")
    
    return {"perplexity": perplexity, "lora_dir": lora_dir}


# ==================== FFT 4BIT на dataset.txt ====================
# Вставь этот код в RUN_ALL_LIB.py после run_full_finetune или в конец файла

from peft import prepare_model_for_kbit_training


def load_fft_txt_corpus(
    txt_path: str = "dataset.txt",
    chunk_size: int = 1500,
    overlap: int = 150,
    val_split: float = 0.05,
):
    with open(txt_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(full_text), step):
        chunk = full_text[i:i + chunk_size]
        if len(chunk.strip()) >= 100:
            chunks.append(chunk)

    train_chunks, val_chunks = train_test_split(
        chunks,
        test_size=val_split,
        random_state=42,
        shuffle=True
    )

    train_dataset = Dataset.from_list([{"text": t} for t in train_chunks])
    val_dataset = Dataset.from_list([{"text": t} for t in val_chunks])

    return train_dataset, val_dataset


# ==================== UNSUPERVISED FFT на dataset.txt ====================

def run_unsupervised_full_finetune(
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    corpus_path: str = "dataset.txt",
    token: str = None,
    num_epoch: int = 3,
    batch_size: int = 1,
    grad_accum_steps: int = 8,
    lr: float = 1e-5,
    max_seq_length: int = 512,
    chunk_size_sym: int = 1500,
    overlap_sym: int = 150,
    val_split: float = 0.05,
    optim: str = "adamw_torch_fused",
    test_status: bool = False,
    make_test = False
):
    print(f"\n===== UNSUPERVISED FULL FINETUNE: {model_name} on {corpus_path} =====")

    train_dataset, val_dataset = load_unsupervised_corpus(
        corpus_path,
        tokenizer=None,
        chunk_size=chunk_size_sym,
        overlap=overlap_sym,
        val_split=val_split
    )

    if test_status:
        train_dataset = train_dataset.select(range(min(10, len(train_dataset))))
        val_dataset = val_dataset.select(range(min(5, len(val_dataset))))

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=token
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        token=token
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.model_max_length = max_seq_length
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,
            return_overflowing_tokens=False
        )

    train_tok = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    val_tok = val_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )

    output_dir = (
        f"FFT_UNSUP_{model_name.replace('/', '_')}"
        f"_dataset_txt_epochs_{num_epoch}"
        f"_maxlen_{max_seq_length}"
        f"_bs_{batch_size}"
        f"_ga_{grad_accum_steps}"
        f"_lr_{lr}"
    )

    args = TrainingArguments(
        output_dir=output_dir,
        bf16=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=lr,
        warmup_ratio=0.03,
        weight_decay=0.01,
        num_train_epochs=num_epoch,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim=optim,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        data_collator=data_collator,
    )

    trainer.train()

    final_dir = output_dir + "_save"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    eval_result = trainer.evaluate()
    eval_loss = eval_result["eval_loss"]
    perplexity = math.exp(eval_loss)

    print(f"Validation Loss: {eval_loss:.4f}")
    print(f"Validation Perplexity: {perplexity:.2f}")

    if make_test:
        try:
            metrics = {
                "perplexity": perplexity,
                "blue": None,
                "lat": None,
                "tok/s": None,
                "tok/j": None,
            }
    
            write_metrics_to_excel(
                EXCEL_PATH,
                model_name,
                "FFT_UNSUP_DATASET_TXT",
                metrics,
                gpu_name=get_gpu_name(),
                path=final_dir,
                quant="bf16",
                test_path=corpus_path
            )
        except Exception as e:
            print(f"Не удалось записать метрики в Excel: {e}")
    
        del model, tokenizer, trainer
        cleanup_cuda()
        mem("after UNSUP FFT cleanup")
    
        return {
            "perplexity": perplexity,
            "eval_loss": eval_loss,
            "final_dir": final_dir,
            "dataset": corpus_path,
            "precision": "bf16"
        }