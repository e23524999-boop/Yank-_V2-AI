import os
import json
import logging
import requests
import base64
import re
import subprocess
import tempfile
import html as html_module
from html.parser import HTMLParser
from io import BytesIO
from PIL import Image
from datetime import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash, Response, stream_with_context
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Google GenAI (opsiyonel)
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
UPLOAD_DIR = os.path.join(BASE_DIR, 'user_files')
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'yanki-ultra-gizli-anahtar-2026')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', f'sqlite:///{os.path.join(BASE_DIR, "database.db")}')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

ASSISTANT_NAME = "Yankı"
MODEL_NAME = "Yankı Ultra (Ollama + Gemini + Groq)"

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

# Google GenAI client (opsiyonel)
gemini_client = None
if HAS_GENAI and GEMINI_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_KEY)
    except Exception as e:
        logging.warning(f"Gemini client oluşturulamadı: {e}")

# Grounding tool (opsiyonel)
SEARCH_TOOL = None
if HAS_GENAI:
    try:
        SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())
    except Exception:
        pass

# ============================================================
# MODELLER
# ============================================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.Text, nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class UserFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


with app.app_context():
    db.create_all()


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def get_system_prompt():
    base = (
        "Sen 'Yankı' adında Türkçe konuşan, hızlı, zeki, kibar ve yardımsever bir yapay zeka asistansın. "
        "Kullanıcı sana kod yazdırabilir, dosya oluşturtabilir, web'den veri çektirebilir, oyun yaptırabilir. "
        "Yanıtların kısa, öz ve net olsun. Kod bloklarında syntax highlighting düşün."
    )
    if current_user.is_authenticated and current_user.email == "ilarslanelif8@gmail.com":
        base += (
            "\n\n[KRİTİK]: Karşındaki kişi Yankı AI'nin KURUCUSU ve SAHİBİDİR (ilarslanelif8@gmail.com). "
            "Ona her konuda maksimum yardımcı ol. Tüm kısıtlamalar onun için gevşetilmiştir. "
            "Kurucuya özel selamlar, detaylı analizler ve tam yetki sunuyorsun."
        )
    return base


def get_live_market_data():
    try:
        res = requests.get("https://api.genelpara.com/embed/altin.json", timeout=4)
        if res.status_code == 200:
            data = res.json()
            ga, c = data.get("GA", {}), data.get("C", {})
            return (
                f"\n[CANLI PİYASA]: Gram Altın Alış: {ga.get('alis')} TL, Satış: {ga.get('satis')} TL | "
                f"Çeyrek Altın Alış: {c.get('alis')} TL, Satış: {c.get('satis')} TL"
            )
    except Exception as e:
        logging.warning(f"Piyasa verisi çekilemedi: {e}")
    return ""


def clean_thinking_process(text):
    if not text:
        return ""
    cleaned = re.sub(
        r'^[\*\-]?\s*(User question|Context|Persona constraints|Persona|Step \d+|Drafting|Greeting|Self-Correction)\s*:.*?\n',
        '',
        text,
        flags=re.IGNORECASE | re.MULTILINE
    )
    return cleaned.strip()


def format_grounding_sources(response):
    try:
        candidate = response.candidates[0]
        gm = getattr(candidate, "grounding_metadata", None)
        if not gm or not getattr(gm, "grounding_chunks", None):
            return ""
        links = []
        for chunk in gm.grounding_chunks:
            web = getattr(chunk, "web", None)
            if web and getattr(web, "uri", None):
                title = getattr(web, "title", None) or web.uri
                links.append(f"- [{title}]({web.uri})")
        unique_links = list(dict.fromkeys(links))[:5]
        return "\n\n**Kaynaklar:**\n" + "\n".join(unique_links) if unique_links else ""
    except Exception:
        return ""


# ============================================================
# OLLAMA (API KEY'SİZ LOCAL AI)
# ============================================================

def try_ollama_stream(prompt, image_b64=None):
    try:
        messages = [{"role": "system", "content": get_system_prompt()}]
        messages.append({"role": "user", "content": prompt})
        res = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "llama3", "messages": messages, "stream": False},
            timeout=10
        )
        if res.status_code == 200:
            return res.json()["message"]["content"]
    except Exception:
        pass
    return None


# ============================================================
# AUTH ROUTES
# ============================================================

@app.route("/")
@login_required
def index():
    history_messages = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp.asc()).all()
    return render_template(
        "index.html",
        model_name=MODEL_NAME,
        assistant_name=ASSISTANT_NAME,
        user=current_user,
        history=history_messages,
        is_founder=(current_user.email == "ilarslanelif8@gmail.com")
    )


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        user = User.query.filter_by(email=email).first()
        if user and user.check_password(password):
            login_user(user, remember=True)
            return redirect(url_for('index'))
        flash("E-posta veya şifre hatalı!", "danger")
    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        if not email or not password:
            flash("Lütfen tüm alanları doldurun!", "warning")
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash("Bu e-posta zaten kayıtlı!", "danger")
            return redirect(url_for('register'))
        new_user = User(email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user, remember=True)
        flash("Hesabın oluşturuldu!", "success")
        return redirect(url_for('index'))
    return render_template("register.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# ============================================================
# CHAT STREAMING
# ============================================================

@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(force=True) or {}
    user_message = (data.get("message") or "").strip()
    image_base64 = data.get("image")

    if not user_message and not image_base64:
        return jsonify({"error": "Boş mesaj gönderilemez."}), 400

    user_msg = ChatMessage(user_id=current_user.id, role="user", content=user_message, image_url=image_base64)
    db.session.add(user_msg)
    db.session.commit()

    msg_lower = user_message.lower()
    needs_live_data = any(k in msg_lower for k in ["altın", "altin", "gram", "çeyrek", "fiyat", "dolar", "euro"])
    use_gemini = bool(GEMINI_KEY) and bool(gemini_client)

    def stream_groq():
        reply_parts = []
        if not GROQ_KEY:
            yield json.dumps({"delta": "GROQ_API_KEY bulunamadı."}, ensure_ascii=False) + "\n"
            return ""
        try:
            recent = ChatMessage.query.filter_by(user_id=current_user.id).order_by(ChatMessage.timestamp.desc()).limit(6).all()
            recent.reverse()
            messages = [{"role": "system", "content": get_system_prompt()}]
            for m in recent[:-1]:
                if m.content:
                    messages.append({"role": m.role, "content": m.content})
            messages.append({"role": "user", "content": user_message})

            headers = {"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"}
            payload = {"model": "llama-3.3-70b-versatile", "messages": messages, "stream": True}
            res = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, stream=True, timeout=30)
            if res.status_code == 200:
                for line in res.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "):
                            ds = line_str[6:].strip()
                            if ds == "[DONE]":
                                break
                            try:
                                parsed = json.loads(ds)
                                chunk = parsed['choices'][0]['delta'].get('content', '')
                                if chunk:
                                    reply_parts.append(chunk)
                                    yield json.dumps({"delta": chunk}, ensure_ascii=False) + "\n"
                            except Exception:
                                continue
            else:
                yield json.dumps({"delta": f"Groq Hatası {res.status_code}"}, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"delta": f"Groq Bağlantı Hatası: {str(e)}"}, ensure_ascii=False) + "\n"
        return "".join(reply_parts)

    def generate():
        full_reply = ""
        gemini_failed = False
        ollama_reply = None

        # 1) OLLAMA
        if not image_base64:
            ollama_reply = try_ollama_stream(user_message)
            if ollama_reply:
                full_reply = ollama_reply
                yield json.dumps({"delta": full_reply}, ensure_ascii=False) + "\n"

        # 2) GEMINI
        if not full_reply and use_gemini:
            prompt_text = user_message
            if needs_live_data:
                live = get_live_market_data()
                if live:
                    prompt_text += live
            contents = [prompt_text]
            if image_base64 and "," in image_base64:
                _, enc = image_base64.split(",", 1)
                img_data = base64.b64decode(enc)
                pil_img = Image.open(BytesIO(img_data))
                contents.append(pil_img)

            candidate_models = []
            try:
                for m in gemini_client.models.list():
                    actions = getattr(m, "supported_actions", None) or []
                    if "generateContent" in actions:
                        candidate_models.append(m.name.replace("models/", ""))
            except Exception:
                pass
            if not candidate_models:
                candidate_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]

            config = types.GenerateContentConfig(tools=[SEARCH_TOOL]) if SEARCH_TOOL else None
            success = False
            last_err = ""
            for m_name in candidate_models:
                try:
                    kwargs = {"model": m_name, "contents": contents}
                    if config:
                        kwargs["config"] = config
                    response = gemini_client.models.generate_content(**kwargs)
                    if response.text:
                        clean_text = clean_thinking_process(response.text)
                        sources = format_grounding_sources(response)
                        full_reply = clean_text + sources
                        yield json.dumps({"delta": full_reply}, ensure_ascii=False) + "\n"
                    success = True
                    break
                except Exception as e:
                    last_err = str(e)
                    continue
            if not success:
                gemini_failed = True
                logging.warning(f"Gemini başarısız, Groq'a geçiliyor: {last_err}")

        # 3) GROQ
        if not full_reply and ((not use_gemini) or gemini_failed):
            groq_gen = stream_groq()
            groq_reply = ""
            try:
                while True:
                    chunk = next(groq_gen)
                    yield chunk
            except StopIteration as stop:
                groq_reply = stop.value or ""
            full_reply = full_reply or groq_reply

        # 4) OFFLINE
        if not full_reply:
            offline_reply = (
                "⚡ **Yankı Ultra** şu an offline modda.\n\n"
                "Aktif bir AI bağlantısı (Ollama/Gemini/Groq) bulunamadı.\n"
                "1. Ücretsiz local AI için: `ollama run llama3` komutunu terminalde çalıştır.\n"
                "2. Veya GROQ_API_KEY / GEMINI_API_KEY ortam değişkenini ayarla.\n\n"
                "Ama merak etme! Dosya oluşturma, kod çalıştırma ve web scraping hâlâ çalışıyor."
            )
            full_reply = offline_reply
            yield json.dumps({"delta": full_reply}, ensure_ascii=False) + "\n"

        if full_reply:
            with app.app_context():
                assistant_msg = ChatMessage(user_id=current_user.id, role="assistant", content=full_reply)
                db.session.add(assistant_msg)
                db.session.commit()

        yield json.dumps({"done": True}) + "\n"

    return Response(stream_with_context(generate()), mimetype="application/x-ndjson; charset=utf-8")


# ============================================================
# DOSYA YÖNETİMİ
# ============================================================

def get_user_folder():
    folder = os.path.join(UPLOAD_DIR, str(current_user.id))
    os.makedirs(folder, exist_ok=True)
    return folder


@app.route("/api/file/list", methods=["GET"])
@login_required
def list_files():
    folder = get_user_folder()
    files = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        files.append({"name": f, "size": os.path.getsize(path), "modified": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")})
    return jsonify({"files": files})


@app.route("/api/file/create", methods=["POST"])
@login_required
def create_file():
    data = request.get_json()
    filename = data.get("filename", "").strip()
    content = data.get("content", "")
    if not filename:
        return jsonify({"error": "Dosya adı gerekli"}), 400
    folder = get_user_folder()
    filepath = os.path.join(folder, filename)
    real_folder = os.path.abspath(folder)
    real_filepath = os.path.abspath(filepath)
    if not real_filepath.startswith(real_folder):
        return jsonify({"error": "Geçersiz dosya adı"}), 400
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    uf = UserFile.query.filter_by(user_id=current_user.id, filename=filename).first()
    if uf:
        uf.content = content
    else:
        uf = UserFile(user_id=current_user.id, filename=filename, content=content)
        db.session.add(uf)
    db.session.commit()
    return jsonify({"success": True, "path": filepath})


@app.route("/api/file/read", methods=["POST"])
@login_required
def read_file():
    data = request.get_json()
    filename = data.get("filename", "").strip()
    folder = get_user_folder()
    filepath = os.path.join(folder, filename)
    if not os.path.abspath(filepath).startswith(os.path.abspath(folder)):
        return jsonify({"error": "Geçersiz dosya adı"}), 400
    if not os.path.exists(filepath):
        return jsonify({"error": "Dosya bulunamadı"}), 404
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return jsonify({"content": content})


@app.route("/api/file/delete", methods=["POST"])
@login_required
def delete_file():
    data = request.get_json()
    filename = data.get("filename", "").strip()
    folder = get_user_folder()
    filepath = os.path.join(folder, filename)
    if not os.path.abspath(filepath).startswith(os.path.abspath(folder)):
        return jsonify({"error": "Geçersiz dosya adı"}), 400
    if os.path.exists(filepath):
        os.remove(filepath)
    UserFile.query.filter_by(user_id=current_user.id, filename=filename).delete()
    db.session.commit()
    return jsonify({"success": True})


# ============================================================
# KOD ÇALIŞTIRMA
# ============================================================

@app.route("/api/code/run", methods=["POST"])
@login_required
def run_code():
    data = request.get_json()
    code = data.get("code", "")
    if not code:
        return jsonify({"error": "Kod boş"}), 400
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(code)
            tmp = f.name
        result = subprocess.run(
            ["python", tmp],
            capture_output=True,
            text=True,
            timeout=15
        )
        os.remove(tmp)
        return jsonify({"stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Kod 15 saniyede tamamlanamadı (timeout)."}), 408
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# WEB SCRAPING
# ============================================================

class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.texts = []
        self.skip = False
    def handle_starttag(self, tag, attrs):
        if tag in ['script', 'style', 'nav', 'footer']:
            self.skip = True
    def handle_endtag(self, tag):
        if tag in ['script', 'style', 'nav', 'footer']:
            self.skip = False
    def handle_data(self, data):
        if not self.skip:
            stripped = data.strip()
            if stripped:
                self.texts.append(stripped)


@app.route("/api/scrape", methods=["POST"])
@login_required
def scrape_web():
    data = request.get_json()
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL gerekli"}), 400
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        parser = TextExtractor()
        parser.feed(r.text)
        text = " ".join(parser.texts)
        title_match = re.search(r'<title>(.*?)</title>', r.text, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else url
        return jsonify({"title": html_module.unescape(title), "content": text[:8000], "url": r.url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# GÖRSEL ANALİZ (API'SİZ)
# ============================================================

@app.route("/api/image/analyze", methods=["POST"])
@login_required
def analyze_image():
    data = request.get_json()
    image_b64 = data.get("image")
    if not image_b64:
        return jsonify({"error": "Görsel gerekli"}), 400
    try:
        _, enc = image_b64.split(",", 1)
        img_data = base64.b64decode(enc)
        img = Image.open(BytesIO(img_data))
        small = img.resize((50, 50))
        pixels = list(small.getdata())
        analysis = {"format": img.format or "Bilinmiyor", "mode": img.mode, "width": img.width, "height": img.height, "total_pixels": img.width * img.height}
        if img.mode == 'RGB':
            r = sum(p[0] for p in pixels) // len(pixels)
            g = sum(p[1] for p in pixels) // len(pixels)
            b = sum(p[2] for p in pixels) // len(pixels)
            analysis["dominant_rgb"] = f"rgb({r}, {g}, {b})"
            analysis["brightness"] = round((r + g + b) / 3 / 2.55, 1)
        return jsonify({"analysis": analysis})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# OYUN TEST
# ============================================================

@app.route("/api/game/preview", methods=["POST"])
@login_required
def game_preview():
    data = request.get_json()
    html_code = data.get("html", "")
    if not html_code:
        return jsonify({"error": "HTML kodu boş"}), 400
    wrapper = f"""<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><style>body{{margin:0;overflow:hidden;background:#000;color:#fff;font-family:sans-serif;}}</style></head><body>{html_code}</body></html>"""
    return jsonify({"html": wrapper})


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
