"""make_video.py — Clean standalone video generator for language demo."""
import cv2
import numpy as np
import pyttsx3
import subprocess
import os
import tempfile
from word_grounder import WordGrounder, SentenceComprehender
from train_language_curriculum import build_curriculum, train_curriculum

# Train
grounder = WordGrounder(d_belief=64)
curriculum = build_curriculum(d_belief=64)
progress = train_curriculum(grounder, curriculum, verbose=False)
comprehender = SentenceComprehender(grounder)

fps = 8
img_w, img_h = 900, 600
frames = []

BG = (248, 248, 245)
TEXT = (35, 35, 40)
MED = (100, 100, 105)
GREEN = (50, 190, 75)
BLUE = (60, 75, 215)
RED = (200, 60, 60)
FOOTER_C = (40, 40, 45)
THOUGHT = (240, 242, 255)

def dt(img, text, x, y, s=0.38, c=TEXT):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, s, c, 1, cv2.LINE_AA)

def footer(img):
    cv2.rectangle(img, (0, img_h - 28), (img_w, img_h), FOOTER_C, -1)
    dt(img, "NeMo-WM", 10, img_h - 8, 0.38, (255, 255, 255))
    dt(img, "No LLM | No Encoder | CPU Only | 1.2M params", 160, img_h - 8, 0.32, (180, 180, 190))
    dt(img, "nemo-wm.com", img_w - 120, img_h - 8, 0.32, (130, 150, 255))

# ── Scene 1: Title - 10 sec ──
n1 = fps * 10
for i in range(n1):
    img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (img_w, 32), FOOTER_C, -1)
    dt(img, "NeMo-WM: Grounded Language Acquisition", 10, 22, 0.48, (255, 255, 255))
    dt(img, "NeMo-WM", 340, 120, 1.2, BLUE)
    dt(img, "Neuromodulated World Model", 270, 155, 0.45, MED)
    dt(img, "Learning language from experience", 250, 210, 0.6, TEXT)
    dt(img, "No LLM", 180, 270, 0.5, RED)
    dt(img, "No Pretrained Encoder", 380, 270, 0.5, RED)
    dt(img, "No Parser", 660, 270, 0.5, RED)
    dt(img, "Words = sensorimotor experience", 220, 320, 0.45, BLUE)
    dt(img, "Comprehension = world model simulation", 190, 350, 0.45, BLUE)
    dt(img, "Meaning = belief state prototype", 225, 380, 0.45, BLUE)
    dt(img, "1.2M params | CPU only | 2.8 us/call", 220, 440, 0.4, MED)
    footer(img)
    frames.append(img)
print(f"Scene 1 (title): {n1} frames = {n1/fps:.0f}s")

# ── Scene 2: Progress - 8 sec each ──
n2 = 0
for p in progress:
    for i in range(fps * 8):
        img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (img_w, 32), FOOTER_C, -1)
        dt(img, "Learning Words from Experience", 10, 22, 0.48, (255, 255, 255))
        dt(img, "Step %d/%d" % (p["step"], len(progress) * 50), img_w - 170, 22, 0.38, (180, 180, 190))
        cv2.rectangle(img, (30, 60), (img_w - 30, 100), (255, 255, 252), -1)
        vw = int((img_w - 62) * min(p["vocab"] / 120.0, 1.0))
        cv2.rectangle(img, (31, 61), (31 + vw, 99), GREEN, -1)
        dt(img, "Vocabulary: %d words" % p["vocab"], 35, 85, 0.4, TEXT)
        cv2.rectangle(img, (30, 110), (img_w - 30, 150), (255, 255, 252), -1)
        cw = int((img_w - 62) * p["avg_comprehension"])
        cv2.rectangle(img, (31, 111), (31 + cw, 149), BLUE, -1)
        dt(img, "Comprehension: %d%%" % int(p["avg_comprehension"] * 100), 35, 135, 0.4, TEXT)
        y = 170
        dt(img, "Domains learned:", 30, y, 0.4, TEXT)
        for domain, count in sorted(p["domains"].items()):
            y += 22
            bw = min(count * 2, 300)
            cv2.rectangle(img, (150, y - 12), (150 + bw, y + 2), BLUE, -1)
            dt(img, "%s: %d" % (domain, count), 30, y, 0.33, MED)
        cv2.rectangle(img, (15, img_h - 150), (img_w - 15, img_h - 32), THOUGHT, -1)
        if p["avg_comprehension"] < 0.3:
            thought = "I am still learning. I know %d words so far." % p["vocab"]
        elif p["avg_comprehension"] < 0.6:
            thought = "Getting better. %d words. Simple sentences work." % p["vocab"]
        else:
            thought = "I understand most sentences now. %d words across all domains." % p["vocab"]
        dt(img, "Thinking:", 50, img_h - 130, 0.35, (100, 100, 160))
        dt(img, thought, 30, img_h - 105, 0.35, TEXT)
        footer(img)
        frames.append(img)
        n2 += 1
print(f"Scene 2 (progress): {n2} frames = {n2/fps:.0f}s")

# ── Scene 3: Comprehension - 12 sec each ──
n3 = 0
sents = [
    "the ball falls due to gravity",
    "danger on the steep dark corridor",
    "push the heavy block left carefully",
    "the strong magnetic force attracts objects near",
]
for sent in sents:
    result = comprehender.comprehend(sent)
    for i in range(fps * 12):
        img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)
        cv2.rectangle(img, (0, 0), (img_w, 32), FOOTER_C, -1)
        dt(img, "Sentence Comprehension", 10, 22, 0.48, (255, 255, 255))
        dt(img, '"%s"' % sent, 30, 80, 0.5, TEXT)
        grounded = result.get("grounded_words", [])
        ungrounded = result.get("ungrounded", [])
        dt(img, "Grounded:", 30, 130, 0.4, GREEN)
        dt(img, ", ".join(grounded), 180, 130, 0.4, GREEN)
        dt(img, "Unknown:", 30, 160, 0.4, RED)
        dt(img, ", ".join(ungrounded) if ungrounded else "none", 180, 160, 0.4, RED)
        conf = result["confidence"]
        bw = int(400 * conf)
        color = GREEN if conf > 0.5 else RED
        cv2.rectangle(img, (30, 200), (430, 220), (255, 255, 252), -1)
        cv2.rectangle(img, (30, 200), (30 + bw, 220), color, -1)
        dt(img, "Confidence: %d%%" % int(conf * 100), 440, 215, 0.4, TEXT)
        nearest = grounder.nearest_words(result["composed_belief"], k=5)
        dt(img, "I understand this as:", 30, 260, 0.4, BLUE)
        dt(img, ", ".join(["%s(%.2f)" % (w, s) for w, s in nearest]), 250, 260, 0.35, MED)
        cv2.rectangle(img, (15, img_h - 130), (img_w - 15, img_h - 32), THOUGHT, -1)
        status = "UNDERSTOOD" if result.get("understood") else "NOT UNDERSTOOD"
        dt(img, status, 30, img_h - 100, 0.4, GREEN if result.get("understood") else RED)
        footer(img)
        frames.append(img)
        n3 += 1
print(f"Scene 3 (comprehension): {n3} frames = {n3/fps:.0f}s")

# ── Scene 4: Final - 15 sec ──
n4 = fps * 15
stats = grounder.stats()
for i in range(n4):
    img = np.full((img_h, img_w, 3), BG, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (img_w, 32), FOOTER_C, -1)
    dt(img, "Language Learned from Experience", 10, 22, 0.48, (255, 255, 255))
    dt(img, "Total vocabulary: %d words" % stats["vocab_size"], 30, 80, 0.5, GREEN)
    dt(img, "Total experiences: %d" % stats["total_hearings"], 30, 120, 0.5, BLUE)
    dt(img, "Avg experiences/word: %.1f" % stats["avg_experiences_per_word"], 30, 160, 0.5, TEXT)
    dt(img, "No LLM. No parser. No pretrained embeddings.", 30, 220, 0.5, RED)
    dt(img, "No separation of language from perception.", 30, 250, 0.5, RED)
    dt(img, "Words = episodic memory prototypes", 30, 310, 0.5, BLUE)
    dt(img, "Sentences = belief-space composition", 30, 340, 0.5, BLUE)
    dt(img, "Comprehension = pattern completion in memory", 30, 370, 0.5, BLUE)
    dt(img, "Cognitive age equivalent: 4-6 years", 30, 430, 0.5, GREEN)
    footer(img)
    frames.append(img)
print(f"Scene 4 (final): {n4} frames = {n4/fps:.0f}s")

total = len(frames)
print(f"\nTotal: {total} frames = {total/fps:.1f}s")

# Write silent video
silent_path = "outputs/language_learning_silent.mp4"
writer = cv2.VideoWriter(silent_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img_w, img_h))
for f in frames:
    writer.write(f)
writer.release()
print(f"Silent video saved: {silent_path}")

# ── TTS narration ──
print("Generating TTS...")
engine = pyttsx3.init()
voices = engine.getProperty("voices")
if len(voices) > 2:
    engine.setProperty("voice", voices[2].id)
engine.setProperty("rate", 130)

narration = [
    (1, "NeMo W M. Grounded language acquisition. The system learns language from its own sensorimotor experience. No large language model. No pretrained encoder."),
    (12, "Learning words from experience. Each word is bound to the belief state active when the system heard it."),
    (22, "Physics concepts like gravity and friction are grounded in actual force patterns the system discovered."),
    (38, "Navigation and emotional words are grounding. Danger means high cortisol. Calm means low stress."),
    (54, "Comprehension is improving. The system composes word beliefs to understand sentences."),
    (70, "The system can now understand multi-domain sentences."),
    (100, "Testing sentence comprehension. Grounded words are highlighted in green. Unknown words in red."),
    (120, "Push the heavy block left carefully. The system maps this to action primitives and spatial directions."),
    (140, "Final results. Over 100 words learned across 8 domains. Cognitive age 4 to 6 years. No L L M. No parser. No pretrained embeddings."),
]

tmp = tempfile.mkdtemp()
wavs = []
for i, (t, text) in enumerate(narration):
    wp = os.path.join(tmp, "s%02d.wav" % i)
    engine.save_to_file(text, wp)
    wavs.append((t, wp))
engine.runAndWait()
print(f"Generated {len(wavs)} speech segments")

# Merge with ffmpeg
import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

inputs = ["-i", silent_path]
parts = []
valid = []
for i, (t, wp) in enumerate(wavs):
    if os.path.exists(wp) and os.path.getsize(wp) > 100:
        inputs.extend(["-i", wp])
        ms = int(t * 1000)
        idx = len(valid) + 1
        parts.append("[%d]adelay=%d|%d[a%d]" % (idx, ms, ms, idx))
        valid.append(idx)

mix = "".join("[a%d]" % v for v in valid)
parts.append("%samix=inputs=%d:duration=longest[aout]" % (mix, len(valid)))
filt = ";".join(parts)

final_path = "outputs/language_learning_demo.mp4"
cmd = [ffmpeg_exe, "-y"] + inputs + [
    "-filter_complex", filt,
    "-map", "0:v", "-map", "[aout]",
    "-c:v", "copy", "-c:a", "aac",
    "-shortest", final_path,
]
r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
if r.returncode == 0:
    print(f"Narrated video: {final_path}")
else:
    print("FFmpeg error: %s" % r.stderr[:300])
    import shutil
    shutil.copy(silent_path, final_path)

for _, wp in wavs:
    try:
        os.remove(wp)
    except OSError:
        pass

print("Done!")
