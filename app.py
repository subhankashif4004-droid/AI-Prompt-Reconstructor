"""
AI Prompt Reconstructor & Enhancer
"""

from flask import Flask, render_template, request, jsonify
import re
import json
import os
import requests
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()

# SMART IMPORT
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YT_AVAILABLE = True
except ImportError:
    YT_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

app = Flask(__name__)

# ============================================
# CONFIGURATION
# ============================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

def configure_genai(api_key=None):
    """
    Configures the Gemini API. 
    Returns True if library configuration succeeded.
    """
    global AI_MODE
    key_to_use = api_key or os.environ.get("GEMINI_API_KEY")
    
    # Try the library first if available
    if GEMINI_AVAILABLE and key_to_use:
        try:
            import google.generativeai as genai
            genai.configure(api_key=key_to_use)
            model = genai.GenerativeModel('gemini-1.5-flash')
            AI_MODE = "gemini"
            return True
        except Exception:
            pass
    
    # If library fails/unavailable but we have a key, we'll use REST in the analyze call
    if key_to_use:
        AI_MODE = "gemini-rest"
        return True
        
    AI_MODE = "builtin"
    return False


def gemini_call_rest(prompt, api_key):
    """Direct REST call to Gemini API for Python 3.14 compatibility."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 2048,
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        else:
            raise Exception(f"API Error {response.status_code}: {response.text}")
    except Exception as e:
        raise Exception(f"REST call failed: {str(e)}")

# Initial configuration
configure_genai(GEMINI_API_KEY)


# ============================================
# YOUTUBE TRANSCRIPT - WORKS WITH ALL VERSIONS
# ============================================
def extract_video_id(url):
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com\/live\/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url.strip()):
        return url.strip()
    return None


def get_transcript(video_id):
    """
    Fetch transcript - Compatible with ALL versions of youtube-transcript-api
    Tries multiple methods to ensure it works.
    """
    if not YT_AVAILABLE:
        return {
            'success': False,
            'error': 'youtube-transcript-api is not installed. Run: pip install youtube-transcript-api'
        }

    # =============================================
    # METHOD 1: New API (v1.0.0+) - 2024/2025
    # =============================================
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_data = ytt_api.fetch(video_id)

        # New API returns a Transcript object
        snippets = []
        total_duration = 0

        # Handle different response formats
        if hasattr(transcript_data, 'snippets'):
            # Newest format with snippets attribute
            for snippet in transcript_data.snippets:
                snippets.append({
                    'text': snippet.text if hasattr(snippet, 'text') else str(snippet),
                    'start': snippet.start if hasattr(snippet, 'start') else 0,
                    'duration': snippet.duration if hasattr(snippet, 'duration') else 0
                })
        elif hasattr(transcript_data, '__iter__'):
            # Iterable format
            for entry in transcript_data:
                if isinstance(entry, dict):
                    snippets.append(entry)
                elif hasattr(entry, 'text'):
                    snippets.append({
                        'text': entry.text,
                        'start': getattr(entry, 'start', 0),
                        'duration': getattr(entry, 'duration', 0)
                    })
                else:
                    snippets.append({'text': str(entry), 'start': 0, 'duration': 0})

        if snippets:
            full_text = ' '.join([s.get('text', '') for s in snippets])
            full_text = full_text.strip()
            if full_text:
                last = snippets[-1]
                total_duration = round(
                    last.get('start', 0) + last.get('duration', 0)
                )
                return {
                    'success': True,
                    'transcript': full_text,
                    'segments': len(snippets),
                    'duration': total_duration,
                    'method': 'new_api_fetch'
                }

        # Try converting to string directly
        text_str = str(transcript_data)
        if text_str and len(text_str) > 20:
            return {
                'success': True,
                'transcript': text_str,
                'segments': 1,
                'duration': 0,
                'method': 'new_api_str'
            }

    except AttributeError:
        pass  # Method doesn't exist, try next
    except Exception as e:
        error_msg_1 = str(e)
        # Continue to next method

    # =============================================
    # METHOD 2: Classic API (v0.6.x) - get_transcript
    # =============================================
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id, languages=['en']
        )
        full_text = ' '.join([entry['text'] for entry in transcript_list])
        if full_text.strip():
            return {
                'success': True,
                'transcript': full_text,
                'segments': len(transcript_list),
                'duration': round(
                    transcript_list[-1]['start'] + 
                    transcript_list[-1]['duration']
                ) if transcript_list else 0,
                'method': 'classic_get_transcript'
            }
    except AttributeError:
        pass
    except Exception as e:
        error_msg_2 = str(e)

    # =============================================
    # METHOD 3: List transcripts and fetch first available
    # =============================================
    try:
        # New API style
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)

        for transcript in transcript_list:
            try:
                fetched = transcript.fetch()
                snippets = []
                if hasattr(fetched, 'snippets'):
                    for s in fetched.snippets:
                        snippets.append(s.text if hasattr(s, 'text') else str(s))
                elif hasattr(fetched, '__iter__'):
                    for entry in fetched:
                        if isinstance(entry, dict):
                            snippets.append(entry.get('text', ''))
                        elif hasattr(entry, 'text'):
                            snippets.append(entry.text)
                        else:
                            snippets.append(str(entry))

                full_text = ' '.join(snippets).strip()
                if full_text:
                    return {
                        'success': True,
                        'transcript': full_text,
                        'segments': len(snippets),
                        'duration': 0,
                        'language': getattr(transcript, 'language', 'unknown'),
                        'method': 'new_api_list'
                    }
            except Exception:
                continue

    except AttributeError:
        pass
    except Exception:
        pass

    # =============================================
    # METHOD 4: Classic list_transcripts
    # =============================================
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        for transcript in transcript_list:
            try:
                fetched = transcript.fetch()
                full_text = ' '.join(
                    [entry['text'] for entry in fetched]
                )
                if full_text.strip():
                    return {
                        'success': True,
                        'transcript': full_text,
                        'segments': len(fetched),
                        'duration': 0,
                        'language': transcript.language,
                        'method': 'classic_list_transcripts'
                    }
            except Exception:
                continue
    except AttributeError:
        pass
    except Exception:
        pass

    # =============================================
    # METHOD 5: Direct fetch with language codes
    # =============================================
    languages_to_try = [
        ['en'], ['en-US'], ['en-GB'], ['hi'], ['es'], ['fr'], ['de'],
        ['pt'], ['ja'], ['ko'], ['zh'], ['ar'], ['ru']
    ]

    for langs in languages_to_try:
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_data = ytt_api.fetch(video_id, languages=langs)
            
            snippets = []
            if hasattr(transcript_data, 'snippets'):
                for s in transcript_data.snippets:
                    snippets.append(s.text if hasattr(s, 'text') else str(s))
            elif hasattr(transcript_data, '__iter__'):
                for entry in transcript_data:
                    if isinstance(entry, dict):
                        snippets.append(entry.get('text', ''))
                    elif hasattr(entry, 'text'):
                        snippets.append(entry.text)
                    else:
                        snippets.append(str(entry))

            full_text = ' '.join(snippets).strip()
            if full_text:
                return {
                    'success': True,
                    'transcript': full_text,
                    'segments': len(snippets),
                    'duration': 0,
                    'language': langs[0],
                    'method': 'multilang_fetch'
                }
        except Exception:
            continue

    # Try classic API with different languages
    for langs in languages_to_try:
        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id, languages=langs
            )
            full_text = ' '.join(
                [entry['text'] for entry in transcript_list]
            )
            if full_text.strip():
                return {
                    'success': True,
                    'transcript': full_text,
                    'segments': len(transcript_list),
                    'duration': 0,
                    'language': langs[0],
                    'method': 'classic_multilang'
                }
        except Exception:
            continue

    last_error = locals().get('error_msg_1', locals().get('error_msg_2', 'Unknown error'))
    
    return {
        'success': False,
        'error': (
            f'Could not fetch transcript for this video.\n\n'
            f'Detail: {last_error}\n\n'
            f'Possible reasons:\n'
            f'• Subtitles/captions are disabled on this video\n'
            f'• Video is private or age-restricted\n'
            f'• Video has no available transcripts\n'
            f'Try a different video with captions enabled.'
        )
    }


# ============================================
# BUILT-IN AI ENGINE (NO API KEY NEEDED)
# ============================================
class BuiltInAI:
    """Free built-in AI engine - No API key required!"""

    @staticmethod
    def analyze_content(transcript):
        """Analyze transcript content and extract key themes."""
        text_lower = transcript.lower()
        words = text_lower.split()
        word_count = len(words)

        categories = {
            'programming': [
                'code', 'programming', 'python', 'javascript', 'function',
                'variable', 'api', 'database', 'framework', 'developer',
                'software', 'html', 'css', 'react', 'node', 'algorithm',
                'debug', 'compile', 'syntax', 'github', 'repository',
                'frontend', 'backend', 'server', 'deploy', 'coding'
            ],
            'ai_ml': [
                'artificial intelligence', 'machine learning', 'neural',
                'deep learning', 'model', 'training', 'dataset', 'gpt',
                'chatgpt', 'prompt', 'ai', 'transformer', 'nlp',
                'computer vision', 'tensorflow', 'pytorch', 'openai',
                'gemini', 'llm', 'large language model', 'generative'
            ],
            'business': [
                'business', 'marketing', 'revenue', 'startup', 'company',
                'customer', 'strategy', 'growth', 'profit', 'investment',
                'entrepreneur', 'brand', 'sales', 'market', 'product',
                'money', 'income', 'finance', 'budget', 'roi'
            ],
            'tutorial': [
                'step', 'tutorial', 'how to', 'guide', 'learn', 'beginner',
                'first', 'start', 'introduction', 'basics', 'setup',
                'install', 'create', 'build', 'make', 'walkthrough'
            ],
            'design': [
                'design', 'ui', 'ux', 'color', 'layout', 'figma',
                'photoshop', 'creative', 'visual', 'typography', 'logo',
                'graphic', 'interface', 'prototype', 'wireframe'
            ],
            'productivity': [
                'productivity', 'workflow', 'efficient', 'organize', 'time',
                'management', 'tool', 'automate', 'system', 'habit',
                'routine', 'focus', 'goal', 'plan', 'schedule'
            ],
            'education': [
                'learn', 'study', 'course', 'education', 'university',
                'student', 'teach', 'knowledge', 'skill', 'training',
                'certification', 'degree', 'school', 'lesson'
            ],
            'science': [
                'science', 'research', 'experiment', 'theory', 'physics',
                'chemistry', 'biology', 'data', 'analysis', 'hypothesis',
                'discovery', 'scientific', 'evidence', 'quantum'
            ],
            'health': [
                'health', 'fitness', 'exercise', 'diet', 'nutrition',
                'mental health', 'wellness', 'meditation', 'sleep',
                'workout', 'body', 'weight', 'muscle', 'protein'
            ],
            'gaming': [
                'game', 'gaming', 'play', 'player', 'level', 'score',
                'gameplay', 'console', 'pc gaming', 'stream', 'twitch',
                'minecraft', 'fortnite', 'valorant', 'controller'
            ]
        }

        detected = {}
        for category, keywords in categories.items():
            score = sum(text_lower.count(kw) for kw in keywords)
            if score > 0:
                detected[category] = score

        sorted_categories = sorted(
            detected.items(), key=lambda x: x[1], reverse=True
        )

        key_phrases = BuiltInAI._extract_key_phrases(text_lower)
        content_type = BuiltInAI._detect_content_type(text_lower)

        return {
            'categories': sorted_categories[:3],
            'key_phrases': key_phrases[:10],
            'content_type': content_type,
            'word_count': word_count,
            'complexity': 'Advanced' if word_count > 2000
                         else 'Intermediate' if word_count > 800
                         else 'Beginner'
        }

    @staticmethod
    def _extract_key_phrases(text):
        """Extract important phrases from text."""
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'and',
            'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
            'neither', 'each', 'every', 'all', 'any', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'only', 'own',
            'same', 'than', 'too', 'very', 'just', 'because', 'if',
            'when', 'where', 'how', 'what', 'which', 'who', 'whom',
            'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her',
            'it', 'its', 'they', 'them', 'their', 'about', 'up',
            'out', 'then', 'here', 'there', 'also', 'like', 'going',
            'know', 'think', 'get', 'got', 'really', 'right', 'okay',
            'actually', 'well', 'now', 'thing', 'things', 'way',
            'something', 'even', 'want', 'need', 'make', 'see',
            'look', 'come', 'go', 'take', 'let', 'say', 'said',
            'one', 'two', 'much', 'many', 'dont', "don't", "it's",
            "that's", "i'm", "you're", "we're", "they're", "there's",
            'use', 'using', 'used', 'people', 'kind', 'basically',
            'gonna', 'gotta', 'wanna', 'cause', 'stuff', 'back'
        }

        words = re.findall(r'\b[a-z]{3,}\b', text)
        filtered = [w for w in words if w not in stop_words]

        freq = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1

        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:15] if count > 1]

    @staticmethod
    def _detect_content_type(text):
        """Detect the type of content."""
        if any(w in text for w in [
            'step by step', 'how to', 'tutorial', 'guide', 'let me show'
        ]):
            return 'Tutorial/How-To'
        elif any(w in text for w in [
            'review', 'pros and cons', 'compared to', 'versus', 'vs'
        ]):
            return 'Review/Comparison'
        elif any(w in text for w in [
            'top 10', 'top 5', 'best', 'list of', 'number one'
        ]):
            return 'Listicle/Top-N'
        elif any(w in text for w in [
            'interview', 'conversation', 'talk about', 'discuss'
        ]):
            return 'Interview/Discussion'
        elif any(w in text for w in [
            'explain', 'what is', 'understand', 'concept', 'theory'
        ]):
            return 'Educational/Explainer'
        elif any(w in text for w in [
            'news', 'update', 'announced', 'released', 'latest'
        ]):
            return 'News/Update'
        elif any(w in text for w in [
            'vlog', 'day in', 'routine', 'morning', 'lifestyle'
        ]):
            return 'Vlog/Lifestyle'
        else:
            return 'General Content'

    @staticmethod
    def reconstruct_prompts(transcript, analysis):
        """Reconstruct likely prompts that could generate this content."""
        prompts = []
        categories = [c[0] for c in analysis['categories']]
        content_type = analysis['content_type']
        key_phrases = analysis['key_phrases']
        complexity = analysis['complexity']

        primary_topic = ', '.join(key_phrases[:3]) if key_phrases else 'this topic'

        if content_type == 'Tutorial/How-To':
            prompts.append({
                'type': '📝 Tutorial Prompt',
                'prompt': (
                    f"Create a detailed step-by-step tutorial about "
                    f"{primary_topic}. Include practical examples, "
                    f"common pitfalls to avoid, and best practices."
                ),
                'confidence': 92
            })
            prompts.append({
                'type': '🎓 Learning Guide',
                'prompt': (
                    f"Write a comprehensive guide for "
                    f"{'beginners' if complexity == 'Beginner' else 'intermediate developers'} "
                    f"on {primary_topic}. Cover setup, implementation, "
                    f"and real-world applications."
                ),
                'confidence': 85
            })
        elif content_type == 'Review/Comparison':
            prompts.append({
                'type': '⚖️ Comparison Prompt',
                'prompt': (
                    f"Write a detailed comparison and review of "
                    f"{primary_topic}. Include pros, cons, "
                    f"use cases, and recommendations."
                ),
                'confidence': 88
            })
        elif content_type == 'Educational/Explainer':
            prompts.append({
                'type': '🧠 Explainer Prompt',
                'prompt': (
                    f"Explain {primary_topic} in detail. Break down "
                    f"complex concepts into simple terms, use analogies, "
                    f"and provide examples."
                ),
                'confidence': 90
            })
        elif content_type == 'Listicle/Top-N':
            prompts.append({
                'type': '📊 List Prompt',
                'prompt': (
                    f"Create a comprehensive list/ranking about "
                    f"{primary_topic}. Include detailed descriptions, "
                    f"comparisons, and your top recommendations."
                ),
                'confidence': 87
            })
        elif content_type == 'News/Update':
            prompts.append({
                'type': '📰 News Analysis Prompt',
                'prompt': (
                    f"Analyze the latest developments in "
                    f"{primary_topic}. Cover what happened, why it matters, "
                    f"and what to expect next."
                ),
                'confidence': 85
            })
        else:
            prompts.append({
                'type': '📋 Content Prompt',
                'prompt': (
                    f"Create detailed content about {primary_topic}. "
                    f"Cover key aspects, provide insights, and include "
                    f"actionable information."
                ),
                'confidence': 82
            })

        # Category-specific prompts
        if 'programming' in categories:
            prompts.append({
                'type': '💻 Coding Prompt',
                'prompt': (
                    f"Write a programming tutorial about {primary_topic}. "
                    f"Include code examples, explanations, and best "
                    f"practices for clean, efficient code."
                ),
                'confidence': 87
            })

        if 'ai_ml' in categories:
            prompts.append({
                'type': '🤖 AI/ML Prompt',
                'prompt': (
                    f"Explain the AI/ML concepts related to "
                    f"{primary_topic}. Cover the architecture, training "
                    f"process, practical applications, and limitations."
                ),
                'confidence': 86
            })

        if 'business' in categories:
            prompts.append({
                'type': '💼 Business Prompt',
                'prompt': (
                    f"Create a business strategy guide about "
                    f"{primary_topic}. Include market analysis, "
                    f"actionable strategies, and real-world case studies."
                ),
                'confidence': 84
            })

        if 'design' in categories:
            prompts.append({
                'type': '🎨 Design Prompt',
                'prompt': (
                    f"Create a design guide about {primary_topic}. "
                    f"Cover design principles, tools, workflows, "
                    f"and current trends."
                ),
                'confidence': 83
            })

        if 'health' in categories:
            prompts.append({
                'type': '💪 Health & Fitness Prompt',
                'prompt': (
                    f"Create a comprehensive health guide about "
                    f"{primary_topic}. Include scientific evidence, "
                    f"practical tips, and actionable advice."
                ),
                'confidence': 82
            })

        # Core prompt always included
        prompts.append({
            'type': '🔄 Reconstructed Core Prompt',
            'prompt': (
                f"Based on the topic — {primary_topic} — "
                f"create {complexity.lower()}-level content that covers "
                f"the key concepts, practical applications, and important "
                f"details. Content type: {content_type}."
            ),
            'confidence': 80
        })

        return prompts[:5]

    @staticmethod
    def enhance_prompts(prompts, analysis):
        """Enhance reconstructed prompts into powerful versions."""
        enhanced = []
        complexity = analysis['complexity']
        key_phrases = analysis['key_phrases']

        for p in prompts:
            original = p['prompt']

            enhanced_prompt = f"""You are an expert content creator and subject matter specialist.

## ROLE & CONTEXT
Act as a senior expert with 15+ years of experience in this field. Your audience is {complexity.lower()}-level learners who want practical, actionable knowledge.

## MAIN TASK
{original}

## DETAILED REQUIREMENTS
1. **Structure**: Use clear headings, subheadings, and logical flow
2. **Depth**: Provide in-depth analysis with real-world examples
3. **Practicality**: Include actionable steps, code snippets (if applicable), and hands-on exercises
4. **Clarity**: Explain complex concepts using simple analogies
5. **Completeness**: Cover prerequisites, main content, and next steps

## KEY TOPICS TO COVER
{chr(10).join(f'- {phrase.title()}' for phrase in key_phrases[:5])}

## OUTPUT FORMAT
- Start with a compelling introduction
- Use bullet points and numbered lists for clarity
- Include practical examples and case studies
- End with key takeaways and action items
- Add resources for further learning

## QUALITY STANDARDS
- Factually accurate and up-to-date
- Engaging and easy to follow
- Professional yet conversational tone
- Optimized for learning and retention"""

            enhanced.append({
                'type': p['type'],
                'original': original,
                'enhanced': enhanced_prompt,
                'improvement_score': 95
            })

        return enhanced


# ============================================
# GEMINI AI ENGINE (FREE WITH API KEY)
# ============================================
class GeminiAI:
    """Google Gemini AI engine - Free tier available!"""

    @staticmethod
    def process_with_gemini(transcript):
        try:
            analysis_prompt = f"""Analyze this YouTube video transcript and:

1. Identify the main topics and themes
2. Determine the content type (tutorial, review, explainer, etc.)
3. Reconstruct 3-5 likely prompts that could have been used to generate similar content
4. Rate each prompt's confidence (0-100%)

Transcript (first 3000 chars):
{transcript[:3000]}

Respond in this JSON format:
{{
    "content_type": "type here",
    "main_topics": ["topic1", "topic2"],
    "complexity": "Beginner/Intermediate/Advanced",
    "reconstructed_prompts": [
        {{
            "type": "emoji + type name",
            "prompt": "the reconstructed prompt",
            "confidence": 85
        }}
    ],
    "summary": "2-3 sentence summary"
}}"""

            response = model.generate_content(analysis_prompt)
            text = response.text

            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {'success': True, 'data': result}
            else:
                return {'success': False, 'error': 'Could not parse AI response'}

        except Exception as e:
            return {'success': False, 'error': str(e)}

    @staticmethod
    def enhance_with_gemini(prompts_data):
        try:
            prompts_text = "\n".join(
                [f"- {p['prompt']}" for p in prompts_data]
            )

            enhance_prompt = f"""Take these reconstructed prompts and enhance each one into a powerful, detailed prompt that would produce exceptional results with any AI:

Original Prompts:
{prompts_text}

For each prompt, create an enhanced version that includes:
1. Clear role/persona assignment
2. Detailed context and requirements
3. Specific output format instructions
4. Quality standards

Respond in this JSON format:
{{
    "enhanced_prompts": [
        {{
            "type": "emoji + type name",
            "original": "original prompt",
            "enhanced": "the full enhanced prompt",
            "improvement_score": 95
        }}
    ]
}}"""

            response = model.generate_content(enhance_prompt)
            text = response.text

            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return {'success': True, 'data': result}
            else:
                return {'success': False, 'error': 'Could not parse AI response'}

        except Exception as e:
            return {'success': False, 'error': str(e)}


@app.route('/')
def index():
    return render_template('index.html', ai_mode=AI_MODE)


@app.route('/validate_api_key', methods=['POST'])
def validate_api_key():
    """Validate the provided Gemini API key."""
    data = request.json
    api_key = data.get('api_key', '').strip()

    if not api_key:
        return jsonify({'success': False, 'error': 'No API key provided'})

    # Try REST validation first / always if library is broken
    try:
        # Simple validation: list models via REST
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return jsonify({'success': True, 'message': 'API Key is valid (REST connection active)'})
        else:
            return jsonify({'success': False, 'error': f'Invalid API Key: {response.text}'})
    except Exception as e:
        return jsonify({'success': False, 'error': f'Validation failed: {str(e)}'})


@app.route('/enhance', methods=['POST'])
def enhance():
    """Enhance prompts endpoint."""
    data = request.json
    prompts = data.get('prompts', [])
    analysis = data.get('analysis', {})
    api_key = data.get('api_key', '').strip()

    if api_key:
        configure_genai(api_key)

    if not prompts:
        return jsonify({'success': False, 'error': 'No prompts provided'})
    
    # If Gemini is enabled and we have an API key, use it
    if AI_MODE == "gemini":
        enhance_result = GeminiAI.enhance_with_gemini(prompts)
        if enhance_result['success']:
            return jsonify({
                'success': True,
                'enhanced_prompts': enhance_result['data'].get('enhanced_prompts', [])
            })

    # Fallback to Built-in AI
    enhanced = BuiltInAI.enhance_prompts(prompts, analysis)
    return jsonify({
        'success': True,
        'enhanced_prompts': enhanced
    })


@app.route('/analyze', methods=['POST'])
def analyze():
    """Main analysis endpoint."""
    data = request.json
    url = data.get('url', '').strip()
    api_key = data.get('api_key', '').strip()

    if api_key:
        configure_genai(api_key)
    else:
        # Fallback to env key if not already configured correctly
        if AI_MODE == "builtin":
            configure_genai()

    if not url:
        return jsonify({
            'success': False, 
            'error': 'Please enter a YouTube URL'
        })

    # Step 1: Extract Video ID
    video_id = extract_video_id(url)
    if not video_id:
        return jsonify({
            'success': False,
            'error': 'Invalid YouTube URL. Please check and try again.'
        })

    # Step 2: Get Transcript
    transcript_result = get_transcript(video_id)
    if not transcript_result['success']:
        return jsonify({
            'success': False,
            'error': transcript_result['error']
        })

    transcript = transcript_result['transcript']

    # Step 3: Analyze & Reconstruct
    if AI_MODE == "gemini":
        ai_result = GeminiAI.process_with_gemini(transcript)
        if ai_result['success']:
            gemini_data = ai_result['data']
            reconstructed = gemini_data.get('reconstructed_prompts', [])

            enhance_result = GeminiAI.enhance_with_gemini(reconstructed)
            if enhance_result['success']:
                enhanced = enhance_result['data'].get('enhanced_prompts', [])
            else:
                analysis = BuiltInAI.analyze_content(transcript)
                enhanced = BuiltInAI.enhance_prompts(reconstructed, analysis)

            return jsonify({
                'success': True,
                'video_id': video_id,
                'transcript_preview': transcript[:500] + '...',
                'transcript_length': len(transcript),
                'segments': transcript_result.get('segments', 0),
                'ai_mode': 'Google Gemini AI',
                'analysis': {
                    'content_type': gemini_data.get('content_type', 'General'),
                    'main_topics': gemini_data.get('main_topics', []),
                    'complexity': gemini_data.get('complexity', 'Intermediate'),
                    'summary': gemini_data.get('summary', ''),
                },
                'reconstructed_prompts': reconstructed,
                'enhanced_prompts': enhanced
            })

    # Built-in AI Analysis (fallback or default)
    analysis = BuiltInAI.analyze_content(transcript)
    reconstructed = BuiltInAI.reconstruct_prompts(transcript, analysis)
    enhanced = BuiltInAI.enhance_prompts(reconstructed, analysis)

    return jsonify({
        'success': True,
        'video_id': video_id,
        'transcript_preview': transcript[:500] + '...',
        'transcript_length': len(transcript),
        'segments': transcript_result.get('segments', 0),
        'ai_mode': 'Built-in AI Engine (Free)',
        'fetch_method': transcript_result.get('method', 'unknown'),
        'analysis': {
            'content_type': analysis['content_type'],
            'main_topics': analysis['key_phrases'][:5],
            'complexity': analysis['complexity'],
            'categories': [
                {'name': c[0], 'score': c[1]}
                for c in analysis['categories']
            ],
            'word_count': analysis['word_count']
        },
        'reconstructed_prompts': reconstructed,
        'enhanced_prompts': enhanced
    })


# ============================================
# DEBUG ROUTE - Check your library version

# RUN THE APP
# ============================================
if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 AI Prompt Reconstructor & Enhancer")
    print("=" * 60)
    print(f"🤖 AI Mode: {AI_MODE.upper()}")
    print(f"📦 YouTube API: {'✅ Installed' if YT_AVAILABLE else '❌ Not installed'}")

    if YT_AVAILABLE:
        try:
            import youtube_transcript_api
            ver = getattr(youtube_transcript_api, '__version__', 'unknown')
            print(f"📌 YT API Version: {ver}")
        except Exception:
            pass

    if AI_MODE == "builtin":
        print("💡 For better results, set GEMINI_API_KEY environment variable")
        print("🔑 Get free key: https://aistudio.google.com/app/apikey")

    print("🌐 Open: http://localhost:5000")
    print("🔧 Debug: http://localhost:5000/debug")
    print("=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)