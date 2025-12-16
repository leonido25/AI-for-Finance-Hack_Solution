# main_advanced.py
import pandas as pd
import requests
import json
import time
import numpy as np
from tqdm import tqdm
import os
from dotenv import load_dotenv
import re
import ast
import faiss
from sentence_transformers import SentenceTransformer

# Подключаем все переменные из окружения
load_dotenv()

# Подключаем ключи
LLM_API_KEY = os.getenv("LLM_API_KEY")
EMBEDDER_API_KEY = os.getenv("EMBEDDER_API_KEY")

class AdvancedRAGAgent:
    """Продвинутый RAG агент с FAISS и системой валидации"""
    
    def __init__(self):
        self.llm_client = self.OpenAIClient(
            LLM_API_KEY, 
            "https://ai-for-finance-hack.up.railway.app"
        )
        self.embedder_client = self.OpenAIClient(
            EMBEDDER_API_KEY,
            "https://ai-for-finance-hack.up.railway.app"
        )
        
        # Данные
        self.documents_data = []
        self.sections_data = []
        self.section_embeddings = []
        self.faiss_index = None
        
        # Модели
        self.generation_models = [
            "openrouter/meta-llama/llama-3-70b-instruct",
            "openrouter/google/gemma-3-27b-it", 
            "openrouter/mistralai/mistral-small-3.2-24b-instruct"
        ]
        
        self.embedding_models = [
            "text-embedding-3-small",
            "text-embedding-ada-002"
        ]
        
        self.current_gen_model = self.generation_models[0]
        self.current_embed_model = self.embedding_models[0]
        
        # Локальная модель для реранкинга (fallback)
        try:
            self.rerank_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        except:
            self.rerank_model = None
    
    class OpenAIClient:
        def __init__(self, api_key, base_url):
            self.api_key = api_key
            self.base_url = base_url
            
        def chat_completion(self, model, messages, max_tokens=1200, temperature=0.1):
            url = f"{self.base_url}/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            for attempt in range(3):
                try:
                    response = requests.post(url, headers=headers, json=data, timeout=120)
                    if response.status_code == 429:
                        wait_time = (attempt + 1) * 30
                        time.sleep(wait_time)
                        continue
                    
                    response.raise_for_status()
                    result = response.json()
                    return result["choices"][0]["message"]["content"]
                        
                except Exception as e:
                    print(f"Попытка {attempt + 1} не удалась: {e}")
                    if attempt < 2:
                        time.sleep(15)
            return None
        
        def get_embedding(self, text, model="text-embedding-3-small"):
            url = f"{self.base_url}/v1/embeddings"
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            if isinstance(text, str):
                input_data = [text]
            else:
                input_data = text
            
            data = {
                "model": model,
                "input": input_data
            }
            
            for attempt in range(3):
                try:
                    response = requests.post(url, headers=headers, json=data, timeout=120)
                    if response.status_code == 429:
                        time.sleep(30)
                        continue
                    
                    response.raise_for_status()
                    result = response.json()
                    return [item["embedding"] for item in result["data"]]
                    
                except Exception as e:
                    print(f"Ошибка эмбеддингов: {e}")
                    if attempt < 2:
                        time.sleep(20)
            return None

    def load_training_data(self, csv_path):
        """Загрузка данных с улучшенной обработкой"""
        print("Загрузка тренировочных данных...")
        try:
            train_data = pd.read_csv(csv_path)
            
            if 'id' not in train_data.columns or 'text' not in train_data.columns:
                print("Не найдены колонки 'id' и 'text'")
                return False
            
            self.documents_data = []
            self.sections_data = []
            
            for _, row in train_data.iterrows():
                document = {
                    'id': row['id'],
                    'text': str(row['text']) if pd.notna(row['text']) else ''
                }
                
                if 'annotation' in row and pd.notna(row['annotation']):
                    document['annotation'] = str(row['annotation'])
                else:
                    document['annotation'] = ''
                
                if 'tags' in row and pd.notna(row['tags']):
                    tags_str = str(row['tags'])
                    try:
                        if tags_str.startswith('[') and tags_str.endswith(']'):
                            document['tags'] = ast.literal_eval(tags_str)
                        else:
                            document['tags'] = [tag.strip() for tag in tags_str.split(',')]
                    except:
                        document['tags'] = [tags_str]
                else:
                    document['tags'] = []
                
                self.documents_data.append(document)
                
                # Улучшенный парсинг с сохранением иерархии
                sections = self.advanced_markdown_parsing(document['text'])
                for section in sections:
                    section_data = {
                        'doc_id': document['id'],
                        'title': section['title'],
                        'content': section['content'],
                        'level': section['level'],
                        'full_text': section['full_text'],
                        'doc_annotation': document['annotation'],
                        'doc_tags': document['tags']
                    }
                    self.sections_data.append(section_data)
            
            print(f"✅ Загружено {len(self.documents_data)} документов, {len(self.sections_data)} разделов")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return False

    def advanced_markdown_parsing(self, text):
        """Продвинутый парсинг markdown с сохранением иерархии"""
        if not text:
            return []
        
        sections = []
        lines = text.split('\n')
        current_section = {'title': '', 'content': [], 'level': 2}
        
        for line in lines:
            line = line.strip()
            if line.startswith('## '):
                # Сохраняем предыдущую секцию
                if current_section['content']:
                    sections.append({
                        'title': current_section['title'],
                        'content': '\n'.join(current_section['content']),
                        'level': current_section['level'],
                        'full_text': f"## {current_section['title']}\n" + '\n'.join(current_section['content'])
                    })
                
                # Новая секция уровня ##
                current_section = {'title': line[3:].strip(), 'content': [], 'level': 2}
                
            elif line.startswith('### '):
                # Сохраняем предыдущую секцию
                if current_section['content']:
                    sections.append({
                        'title': current_section['title'],
                        'content': '\n'.join(current_section['content']),
                        'level': current_section['level'],
                        'full_text': f"## {current_section['title']}\n" + '\n'.join(current_section['content'])
                    })
                
                # Новая секция уровня ###
                current_section = {'title': line[4:].strip(), 'content': [], 'level': 3}
                
            elif line and not line.startswith('#'):
                current_section['content'].append(line)
        
        # Добавляем последнюю секцию
        if current_section['content']:
            sections.append({
                'title': current_section['title'],
                'content': '\n'.join(current_section['content']),
                'level': current_section['level'],
                'full_text': f"## {current_section['title']}\n" + '\n'.join(current_section['content'])
            })
        
        # Если разделов нет, создаем один
        if not sections and text.strip():
            sections.append({
                'title': "Основная информация",
                'content': text,
                'level': 2,
                'full_text': text
            })
        
        return sections

    def build_faiss_index(self):
        """Построение FAISS индекса для быстрого поиска"""
        print("Построение FAISS индекса...")
        
        if not self.sections_data:
            print("Нет данных для индексации")
            return False
        
        for embed_model in self.embedding_models:
            print(f"Генерация эмбеддингов с {embed_model}...")
            self.current_embed_model = embed_model
            
            try:
                # Подготавливаем тексты для эмбеддингов
                search_texts = []
                for section in self.sections_data:
                    search_text = self.create_enhanced_search_text(section)
                    search_texts.append(search_text[:2000])
                
                # Генерируем эмбеддинги батчами
                all_embeddings = []
                batch_size = 20
                
                for i in range(0, len(search_texts), batch_size):
                    batch_texts = search_texts[i:i+batch_size]
                    embeddings = self.embedder_client.get_embedding(batch_texts, embed_model)
                    
                    if embeddings and len(embeddings) == len(batch_texts):
                        all_embeddings.extend(embeddings)
                        print(f"Обработано: {i+len(batch_texts)}/{len(search_texts)}")
                    else:
                        print(f"Ошибка батча {i}")
                        break
                    
                    time.sleep(1)
                
                if len(all_embeddings) == len(search_texts):
                    self.section_embeddings = np.array(all_embeddings).astype('float32')
                    
                    # Создаем FAISS индекс
                    dimension = self.section_embeddings.shape[1]
                    self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner Product для косинусной схожести
                    
                    # Нормализуем векторы для косинусной схожести
                    faiss.normalize_L2(self.section_embeddings)
                    self.faiss_index.add(self.section_embeddings)
                    
                    print(f"✅ FAISS индекс построен: {len(self.section_embeddings)} векторов")
                    return True
                else:
                    print(f"❌ Не удалось сгенерировать все эмбеддинги для {embed_model}")
                    continue
                    
            except Exception as e:
                print(f"❌ Ошибка с моделью {embed_model}: {e}")
                continue
        
        return False

    def create_enhanced_search_text(self, section):
        """Создание улучшенного текста для поиска"""
        parts = []
        
        # Заголовок с весом
        if section.get('title'):
            parts.extend([section['title']] * 2)  # Удваиваем вес заголовка
        
        # Аннотация документа
        if section.get('doc_annotation'):
            parts.append(section['doc_annotation'])
        
        # Теги с весом
        if section.get('doc_tags'):
            tags_text = ' '.join([f"{tag} {tag}" for tag in section['doc_tags']])  # Удваиваем теги
            parts.append(tags_text)
        
        # Ключевые части контента
        if section.get('content'):
            # Извлекаем первые предложения и предложения с ключевыми словами
            content_preview = self.extract_important_content(section['content'])
            parts.append(content_preview)
        
        return " ".join(parts)

    def extract_important_content(self, content, max_length=800):
        """Извлечение наиболее важных частей контента"""
        if not content:
            return ""
        
        sentences = re.split(r'[.!?]+', content)
        important_indicators = [
            'срок', 'день', 'рабочий', 'закон', 'федеральный', 'статья',
            'руб', 'сумма', 'лимит', 'процент', 'обязан', 'должен'
        ]
        
        important_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            
            sentence_lower = sentence.lower()
            if any(indicator in sentence_lower for indicator in important_indicators):
                important_sentences.append(sentence)
        
        if important_sentences:
            return " ".join(important_sentences[:5])  # Не более 5 важных предложений
        else:
            return content[:max_length]

    def hybrid_search(self, query, top_k=5):
        """Гибридный поиск с FAISS и реранкингом"""
        if self.faiss_index is None or len(self.section_embeddings) == 0:
            return self.sections_data[:top_k]
        
        try:
            # Получаем эмбеддинг запроса
            query_embedding = self.embedder_client.get_embedding([query], self.current_embed_model)
            if not query_embedding:
                return self.sections_data[:top_k]
            
            query_vec = np.array(query_embedding[0]).astype('float32').reshape(1, -1)
            faiss.normalize_L2(query_vec)
            
            # Поиск в FAISS
            k = min(top_k * 3, len(self.section_embeddings))  # Ищем больше для реранкинга
            distances, indices = self.faiss_index.search(query_vec, k)
            
            # Собираем результаты
            initial_results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.sections_data) and distance > 0.3:  # Порог схожести
                    section = self.sections_data[idx].copy()
                    section['similarity'] = float(distance)
                    initial_results.append(section)
            
            # Реранкинг с учетом ключевых слов
            reranked_results = self.keyword_reranking(query, initial_results)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            print(f"Ошибка поиска FAISS: {e}")
            return self.sections_data[:top_k]

    def keyword_reranking(self, query, initial_results):
        """Реранкинг результатов на основе ключевых слов"""
        if not initial_results:
            return []
        
        query_words = set(query.lower().split())
        
        for result in initial_results:
            # Создаем объединенный текст для анализа
            analysis_text = ""
            if result.get('title'):
                analysis_text += " " + result['title'].lower()
            if result.get('content'):
                analysis_text += " " + result['content'].lower()
            if result.get('doc_annotation'):
                analysis_text += " " + result['doc_annotation'].lower()
            
            # Вычисляем пересечение
            result_words = set(analysis_text.split())
            common_words = query_words.intersection(result_words)
            
            # Бонус за совпадение ключевых слов
            keyword_bonus = len(common_words) / len(query_words) if query_words else 0
            result['keyword_score'] = keyword_bonus
            result['combined_score'] = result.get('similarity', 0) * 0.7 + keyword_bonus * 0.3
        
        # Сортируем по комбинированному score
        initial_results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        return initial_results

    def create_rich_context(self, similar_sections, question):
        """Создание богатого контекста с максимальной информацией"""
        if not similar_sections:
            return "В базе знаний банка не найдено информации по данному вопросу."
        
        context_parts = ["РЕЛЕВАНТНАЯ ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ БАНКА:\n"]
        
        for i, section in enumerate(similar_sections):
            context_parts.append(f"--- ИСТОЧНИК {i+1} [Релевантность: {section.get('combined_score', 0):.3f}] ---")
            
            if section.get('title'):
                context_parts.append(f"ЗАГОЛОВОК: {section['title']}")
            
            if section.get('doc_annotation'):
                context_parts.append(f"КОНТЕКСТ: {section['doc_annotation']}")
            
            if section.get('content'):
                # Берем полное содержание для важных разделов
                if section.get('combined_score', 0) > 0.6:
                    context_parts.append(f"ПОЛНОЕ СОДЕРЖАНИЕ:\n{section['content']}")
                else:
                    # Или релевантные выдержки
                    excerpts = self.extract_relevant_excerpts(section['content'], question)
                    if excerpts:
                        context_parts.append(f"КЛЮЧЕВЫЕ ВЫДЕРЖКИ:\n{excerpts}")
            
            context_parts.append("")  # Пустая строка для разделения
        
        return "\n".join(context_parts)

    def extract_relevant_excerpts(self, content, question, max_excerpts=5):
        """Извлечение наиболее релевантных выдержек из контента"""
        if not content:
            return ""
        
        paragraphs = re.split(r'\n\s*\n', content)
        question_words = set(question.lower().split())
        
        relevant_paragraphs = []
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            paragraph_lower = paragraph.lower()
            paragraph_words = set(paragraph_lower.split())
            common_words = question_words.intersection(paragraph_words)
            
            if common_words:
                relevance = len(common_words) / len(question_words) if question_words else 0
                if relevance > 0.2:
                    relevant_paragraphs.append((paragraph, relevance))
        
        # Сортируем по релевантности и берем лучшие
        relevant_paragraphs.sort(key=lambda x: x[1], reverse=True)
        
        excerpts = []
        total_length = 0
        for paragraph, relevance in relevant_paragraphs[:max_excerpts]:
            if total_length + len(paragraph) > 1500:  # Ограничение общей длины
                break
            excerpts.append(paragraph)
            total_length += len(paragraph)
        
        return "\n\n".join(excerpts) if excerpts else content[:1000]

    def generate_detailed_answer(self, question, use_rag=True):
        """Генерация детализированного ответа с системой валидации"""
        
        # Получаем контекст
        context = ""
        if use_rag and self.faiss_index is not None:
            similar_sections = self.hybrid_search(question, top_k=4)
            context = self.create_rich_context(similar_sections, question)
        
        # Промпт для детализированных ответов
        system_prompt = f"""Ты - финансовый консультант с глубокими знаниями в области банковского дела, кредитов, инвестиций и защиты прав потребителей.
На основе предоставленного контекста дай подробный, практический и структурированный ответ на вопрос пользователя.
ТРЕБОВАНИЯ К ОТВЕТУ:
1. ОБЯЗАТЕЛЬНО используй информацию из предоставленного контекста
2. Структурируй ответ с помощью заголовков (###), подзаголовков и маркированных списков
3. Указывай КОНКРЕТНЫЕ детали: сроки, суммы, номера законов, названия организаций
4. Будь максимально полезным и информативным
5. Объем ответа: 400-1000 слов
6. Если в контексте есть конкретные цифры или сроки - обязательно их укажи
7. Форматируй ответ как профессиональную консультацию

Пример хорошего ответа:
### Сроки блокировки перевода

В российском законодательстве (Федеральный закон № 115-ФЗ) установлены следующие сроки...

- **Блокировка**: В течение 1 рабочего дня
- **Срок приостановки**: До 10 дней (или 30 дней при проверке)
- **Разблокировка**: Автоматически после проверки"""

        user_prompt = f"""ВОПРОС КЛИЕНТА: {question}

ИНФОРМАЦИЯ ИЗ БАЗЫ ЗНАНИЙ БАНКА: {context}

СФОРМУЛИРУЙ ИСЧЕРПЫВАЮЩИЙ ОТВЕТ, используя предоставленную информацию. Ответ должен быть структурированным, детализированным и максимально полезным:"""

        # Первая попытка генерации
        best_answer = None
        best_score = 0
        
        for gen_model in self.generation_models:
            for attempt in range(2):  # 2 попытки на модель
                try:
                    self.current_gen_model = gen_model
                    
                    answer = self.llm_client.chat_completion(
                        model=gen_model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=1500,
                        temperature=0.1 if attempt == 0 else 0.3  # На второй попытке немного креативности
                    )
                    
                    if answer:
                        score = self.evaluate_answer_quality(answer, question, context)
                        
                        if score > best_score:
                            best_answer = answer
                            best_score = score
                        
                        if score > 0.7:  # Хороший ответ
                            print(f"✅ Качественный ответ от {gen_model} (оценка: {score:.2f})")
                            return answer
                        else:
                            print(f"⚠️  Слабый ответ от {gen_model} (оценка: {score:.2f}), пробуем улучшить...")
                    
                except Exception as e:
                    print(f"❌ Ошибка {gen_model}, попытка {attempt+1}: {e}")
                    continue
        
        # Если не нашли хороший ответ, возвращаем лучший из сгенерированных
        if best_answer and best_score > 0.4:
            print(f"✅ Возвращаем лучший найденный ответ (оценка: {best_score:.2f})")
            return best_answer
        else:
            # Fallback
            return self.create_fallback_answer(question, context)

    def evaluate_answer_quality(self, answer, question, context):
        """Оценка качества ответа по нескольким критериям"""
        if not answer or len(answer.strip()) < 200:
            return 0.0
        
        score = 0.0
        
        # 1. Длина ответа (0-0.2)
        length_score = min(len(answer) / 800, 1.0) * 0.2
        score += length_score
        
        # 2. Структура (0-0.3)
        has_headers = '###' in answer or '**' in answer
        has_lists = '-' in answer or '•' in answer or '1.' in answer
        structure_score = 0.3 if (has_headers and has_lists) else 0.1
        score += structure_score
        
        # 3. Конкретика (0-0.3)
        has_numbers = bool(re.search(r'\d+', answer))  # Цифры
        has_laws = any(word in answer.lower() for word in ['закон', 'фз', 'статья', 'норматив'])
        has_dates = any(word in answer.lower() for word in ['день', 'срок', 'месяц', 'год'])
        specificity_score = (has_numbers + has_laws + has_dates) / 3 * 0.3
        score += specificity_score
        
        # 4. Использование контекста (0-0.2)
        # Проверяем, что ответ не слишком общий
        generic_phrases = [
            'обратитесь в банк', 'свяжитесь с поддержкой', 
            'не могу ответить', 'извините, но'
        ]
        is_generic = any(phrase in answer.lower() for phrase in generic_phrases)
        context_score = 0.2 if not is_generic else 0.05
        score += context_score
        
        return min(score, 1.0)

    def create_fallback_answer(self, question, context):
        """Создание ответа, когда обычная генерация не удалась"""
        fallback_prompt = f"""ВОПРОС: {question}

ИНФОРМАЦИЯ: {context if context else "Информация ограничена"}

Дай развернутый, структурированный ответ. Если информации недостаточно, укажи это, но все равно дай максимально подробную консультацию на основе имеющейся в базе данных информации:"""
        
        for gen_model in self.generation_models:
            try:
                answer = self.llm_client.chat_completion(
                    model=gen_model,
                    messages=[{"role": "user", "content": fallback_prompt}],
                    max_tokens=1000,
                    temperature=0.2
                )
                if answer and len(answer) > 300:
                    return answer
            except:
                continue
        
        return f"""По вопросу "{question}" в базе знаний банка найдена ограниченная информация.

Для получения детальной консультации рекомендуем:
1. Обратиться в отделение вашего банка
2. Позвонить на горячую линию службы поддержки
3. Изучить информацию на официальном сайте Банка России

Мы приносим извинения за возможные неудобства и готовы помочь вам с другими вопросами."""

    def process_questions_advanced(self, questions_csv_path, output_path="submission_advanced.csv"):
        """Обработка вопросов с продвинутой системой"""
        print("Запуск продвинутой обработки вопросов...")
        
        try:
            questions_df = pd.read_csv(questions_csv_path)
            
            if 'Вопрос' in questions_df.columns:
                question_col = 'Вопрос'
            elif 'question' in questions_df.columns:
                question_col = 'question'
            else:
                question_col = questions_df.columns[1]
                print(f"Используем колонку: '{question_col}'")
            
            questions = questions_df[question_col].fillna('').astype(str).tolist()
            answers = []
            quality_scores = []
            
            use_rag = self.faiss_index is not None
            print(f"Режим: {'FAISS RAG включен' if use_rag else 'Без RAG'}")
            
            for i, question in enumerate(tqdm(questions, desc="Генерация ответов")):
                if not question or question.strip() == '':
                    answers.append("Вопрос не предоставлен.")
                    quality_scores.append(0.0)
                    continue
                
                answer = self.generate_detailed_answer(question, use_rag)
                answers.append(answer)
                
                # Оценка качества финального ответа
                quality_score = self.evaluate_answer_quality(answer, question, "")
                quality_scores.append(quality_score)
                
                time.sleep(2)
                
                # Промежуточное сохранение каждые 20 вопросов
                if (i + 1) % 20 == 0:
                    temp_df = questions_df.iloc[:len(answers)].copy()
                    temp_df['Ответы на вопрос'] = answers
                    temp_df['Оценка качества'] = quality_scores
                    temp_df.to_csv(f"temp_advanced_{i+1}.csv", index=False)
                    print(f"✅ Сохранено: {i+1} ответов, средняя оценка: {np.mean(quality_scores):.2f}")
            
            # Финальное сохранение
            result_df = questions_df.copy()
            result_df['Ответы на вопрос'] = answers
            result_df['Оценка качества'] = quality_scores
            result_df.to_csv(output_path, index=False, encoding='utf-8')
            
            print(f"\n🎉 Готово! Результаты сохранены в: {output_path}")
            print(f"📊 Статистика:")
            print(f"   - Средняя оценка качества: {np.mean(quality_scores):.3f}")
            print(f"   - Ответов >500 символов: {sum(1 for a in answers if len(a) > 500)}/{len(answers)}")
            print(f"   - Структурированных ответов: {sum(1 for a in answers if '###' in a)}/{len(answers)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка обработки: {e}")
            return False

def main():
    """Основная функция"""
    print("🚀 Запуск Advanced RAG Agent с FAISS...")
    
    # Проверка файлов
    required_files = ['train_data.csv', 'questions.csv']
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} найден")
        else:
            print(f"❌ {file} не найден")
            return
    
    # Проверка ключей
    if not LLM_API_KEY or not EMBEDDER_API_KEY:
        print("❌ API ключи не найдены")
        return
    
    # Проверка/установка FAISS
    try:
        import faiss
        print("✅ FAISS доступен")
    except ImportError:
        print("❌ FAISS не установлен. Установите: pip install faiss-cpu")
        return
    
    # Инициализация агента
    agent = AdvancedRAGAgent()
    
    # Загрузка данных
    print("\n📁 Загрузка данных...")
    if agent.load_training_data('./train_data.csv'):
        print("🔧 Построение FAISS индекса...")
        if agent.build_faiss_index():
            print("✅ FAISS индекс успешно построен")
        else:
            print("⚠️  Продолжаем без FAISS")
    else:
        print("❌ Ошибка загрузки данных")
        return
    
    # Обработка вопросов
    print("\n❓ Генерация детализированных ответов...")
    success = agent.process_questions_advanced('./questions.csv', 'submission_final.csv')
    
    if success:
        print("\n🎉 Задание выполнено успешно!")
        print("💡 Ответы должны быть детализированными, структурированными и информативными")
    else:
        print("\n💥 Ошибка выполнения")

if __name__ == "__main__":
    main()