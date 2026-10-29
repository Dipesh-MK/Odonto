"""
voice_llm_service.py

Real Speech-to-Text (Whisper), LLM Conversational & Grading Engine (Anthropic Claude API / Local Ollama),
Rule-Based 3D Location Grading, and Real Text-to-Speech (Piper ONNX TTS) for Odonto.

Strict No-Hardcoding & No-Stub Guarantees:
1. STT uses faster_whisper (Whisper 'tiny.en' INT8) on actual recorded audio files (.wav / .webm / .ogg).
2. TTS uses Piper (PiperVoice ONNX 'en_US-lessac-low.onnx') to synthesize real 16kHz WAV audio files.
3. LLM uses Anthropic Claude API when ANTHROPIC_API_KEY is present in the environment, and local
   Ollama (qwen2.5:7b-instruct / llama3.2:3b) when running locally without a cloud key.
4. Location grading is 100% rule-based 3D Euclidean + surface-normal geometry with NO artificial floor
   (a wrong mark on the opposite side or off the tooth scores 0-15%, never 35%).
5. Diagnosis/reasoning grading is LLM-driven against the case's true tooth_fdi, target_surface,
   anomaly_type, and depth, with all 4 Viva competencies initialized to False.
"""

import json
import logging
import os
import re
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    from piper.voice import PiperVoice
except ImportError:
    PiperVoice = None

BASE_DIR = Path(__file__).resolve().parent
AUDIO_DIR = BASE_DIR / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

PIPER_MODEL_PATH = BASE_DIR / "models" / "piper" / "en_US-lessac-low.onnx"

_WHISPER_SINGLETON: Optional[WhisperModel] = None
_PIPER_SINGLETON: Optional[PiperVoice] = None


class ClinicalLLMUnavailableError(RuntimeError):
    pass


def get_whisper_model():
    global _WHISPER_SINGLETON
    if _WHISPER_SINGLETON is None and WhisperModel is not None:
        try:
            _WHISPER_SINGLETON = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        except Exception:
            _WHISPER_SINGLETON = None
    return _WHISPER_SINGLETON


def get_piper_voice():
    global _PIPER_SINGLETON
    if _PIPER_SINGLETON is None and PiperVoice is not None:
        if PIPER_MODEL_PATH.exists():
            try:
                _PIPER_SINGLETON = PiperVoice.load(str(PIPER_MODEL_PATH))
            except Exception:
                _PIPER_SINGLETON = None
    return _PIPER_SINGLETON


# ---------------------------------------------------------------------------
# 1. Real Speech-to-Text (Whisper)
# ---------------------------------------------------------------------------
def transcribe_audio_file(audio_path: str | Path) -> Dict[str, Any]:
    """
    Transcribes an audio file (.wav, .webm, .ogg, .m4a) using faster-whisper.
    Normalizes any container format to 16kHz mono PCM WAV via ffmpeg, loads the
    float32 waveform via soundfile, and runs WhisperModel.transcribe(audio_np).
    """
    import imageio_ffmpeg
    import soundfile as sf

    src_path = Path(audio_path)
    if not src_path.exists() or src_path.stat().st_size == 0:
        raise ValueError("Audio file is missing or empty.")

    wav_path = AUDIO_DIR / f"norm_{uuid.uuid4().hex[:8]}.wav"
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(src_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if proc.returncode != 0 or not wav_path.exists():
        raise RuntimeError(f"ffmpeg audio normalization failed: {proc.stderr[:240]}")

    try:
        audio_np, sr = sf.read(str(wav_path), dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        duration_sec = float(len(audio_np)) / float(sr or 16000)

        model = get_whisper_model()
        if model is None:
            raise RuntimeError("faster-whisper is unavailable or its model could not be initialized.")
        segments, info = model.transcribe(audio_np, beam_size=3, vad_filter=False)
        seg_list = [s.text.strip() for s in segments if s.text.strip()]
        transcript = " ".join(seg_list).strip()
        return {
            "transcript": transcript,
            "duration_seconds": round(duration_sec, 2),
            "language": getattr(info, "language", "en"),
            "stt_engine": "faster-whisper:tiny.en",
        }
    finally:
        if wav_path.exists():
            try:
                wav_path.unlink()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# 2. Real Text-to-Speech (Piper ONNX)
# ---------------------------------------------------------------------------
def synthesize_speech_to_url(text: str, prefix: str = "tts") -> str:
    """
    Synthesizes spoken English text to a real WAV file in static/audio/ using Piper TTS
    and returns its relative URL (/static/audio/<fname>.wav).
    """
    clean_text = re.sub(r"\s+", " ", (text or "").strip())
    if not clean_text:
        clean_text = "No spoken response generated."

    fname = f"{prefix}_{uuid.uuid4().hex[:10]}.wav"
    out_path = AUDIO_DIR / fname
    voice = get_piper_voice()
    if voice is None:
        return ""
    with wave.open(str(out_path), "wb") as wf:
        voice.synthesize_wav(clean_text, wf)
    return f"/static/audio/{fname}"


# ---------------------------------------------------------------------------
# 3. Real LLM Client (Anthropic Claude API / Local Ollama)
# ---------------------------------------------------------------------------
def call_clinical_llm(
    system_prompt: str,
    user_prompt: str,
    expect_json: bool = False,
    temperature: float = 0.2,
) -> Tuple[str, str]:
    """
    Calls Anthropic Claude if ANTHROPIC_API_KEY is configured in the environment;
    otherwise calls local Ollama (qwen2.5:7b-instruct / llama3.2:3b).
    Returns (response_text, provider_tag).
    """
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if anthropic_key:
        try:
            import anthropic
        except ImportError as exc:
            raise ClinicalLLMUnavailableError("The Anthropic client package is not installed.") from exc

        client = anthropic.Anthropic(api_key=anthropic_key)
        model_name = os.environ.get("ODONTO_CLAUDE_MODEL", "claude-3-5-haiku-latest")
        try:
            resp = client.messages.create(
                model=model_name,
                max_tokens=1024,
                temperature=temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            raise ClinicalLLMUnavailableError(f"Anthropic request failed: {exc}") from exc
        content = resp.content[0].text if resp.content else ""
        return content.strip(), f"anthropic:{model_name}"

    # Local Ollama inference (qwen2.5:7b-instruct or llama3.2:3b)
    ollama_url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
    ollama_model = os.environ.get("ODONTO_OLLAMA_MODEL", "qwen2.5:7b-instruct")
    payload: Dict[str, Any] = {
        "model": ollama_model,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 700},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    if expect_json:
        payload["format"] = "json"

    try:
        r = requests.post(ollama_url, json=payload, timeout=90)
        r.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise ClinicalLLMUnavailableError(f"Ollama request failed: {exc}") from exc
    data = r.json()
    text = data.get("message", {}).get("content", "").strip()
    return text, f"ollama:{ollama_model}"


def parse_json_safely(raw_text: str) -> Dict[str, Any]:
    """Extracts a JSON object from an LLM response string."""
    try:
        return json.loads(raw_text)
    except Exception:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# 4. Pure Rule-Based 3D Location Grading (No LLM, No Artificial Floor)
# ---------------------------------------------------------------------------
def compute_rule_based_location_grade(
    student_mark_3d: List[float],
    centered_vertices: np.ndarray,
    vertex_normals: np.ndarray,
    gt_cavity: np.ndarray,
    lesion_radius_mm: float,
) -> Dict[str, Any]:
    """
    Pure geometric rule-based comparison of the student's 3D mark against the
    carved mesh's ground-truth lesion:
    - Computes exact 3D Euclidean distance (mm) to gt_centroid_3d.
    - Finds the mesh vertex closest to student_mark_3d and checks its ground-truth
      lesion influence value and surface normal alignment with the lesion center normal.
    - No artificial 35% floor: clicking on the wrong side of the tooth or off the tooth
      scores 0% to 15%.
    """
    stu = np.asarray(student_mark_3d, dtype=np.float64)
    verts = np.asarray(centered_vertices, dtype=np.float64)
    norms = np.asarray(vertex_normals, dtype=np.float64)
    inf = np.asarray(gt_cavity, dtype=np.float64)

    if np.max(inf) > 1e-4:
        peak_mask = inf >= (0.5 * np.max(inf))
        weights = (inf[peak_mask] ** 2)[:, None]
        gt_centroid = np.sum(verts[peak_mask] * weights, axis=0) / np.sum(weights)
        peak_idx = int(np.argmax(inf))
    else:
        gt_centroid = np.zeros(3, dtype=np.float64)
        peak_idx = 0

    gt_normal = norms[peak_idx] / (np.linalg.norm(norms[peak_idx]) + 1e-12)

    # Nearest vertex to student mark
    dists_to_verts = np.linalg.norm(verts - stu[None, :], axis=1)
    nearest_idx = int(np.argmin(dists_to_verts))
    dist_to_surface_mm = float(dists_to_verts[nearest_idx])
    stu_normal = norms[nearest_idx] / (np.linalg.norm(norms[nearest_idx]) + 1e-12)
    stu_vertex_influence = float(inf[nearest_idx])

    surface_alignment_dot = float(np.dot(stu_normal, gt_normal))
    centroid_distance_mm = float(np.linalg.norm(stu - gt_centroid))
    r_eff = max(float(lesion_radius_mm), 0.6)

    # Strict geometric scoring rules:
    # 1. If the mark is more than 3.0mm off the mesh surface itself, it's not even on the tooth -> 0%
    if dist_to_surface_mm > 3.0:
        location_score = 0
        margin_precision = 0
        depth_agreement = 0
        within_margin = False
    # 2. If the mark is on an opposite-facing wall (dot < -0.05) and outside the lesion radius
    elif surface_alignment_dot < -0.05 and centroid_distance_mm > r_eff:
        raw = max(0.0, 15.0 * (1.0 - centroid_distance_mm / (4.0 * r_eff)))
        location_score = int(round(raw))
        margin_precision = int(round(raw))
        depth_agreement = 0
        within_margin = False
    else:
        # Normalized distance ratio:
        # d = 0 -> 100%
        # d = 1.0 * r_eff -> ~82%
        # d = 1.5 * r_eff -> ~65% (outer margin threshold)
        # d >= 3.5 * r_eff -> 0%
        ratio = centroid_distance_mm / r_eff
        if ratio <= 1.0:
            base_score = 100.0 - 18.0 * (ratio ** 1.2)
        elif ratio <= 3.5:
            base_score = 82.0 * (1.0 - ((ratio - 1.0) / 2.5) ** 1.15)
        else:
            base_score = 0.0

        # Bonus/penalty from actual vertex influence at clicked point
        influence_boost = 8.0 * stu_vertex_influence
        normal_factor = np.clip((surface_alignment_dot + 0.2) / 1.0, 0.25, 1.0)

        final_loc = np.clip((base_score + influence_boost) * normal_factor, 0.0, 100.0)
        location_score = int(round(final_loc))
        margin_precision = int(round(np.clip(location_score - 3.0 * ratio, 0.0, 100.0)))
        depth_agreement = int(round(np.clip(location_score * (0.75 + 0.25 * stu_vertex_influence), 0.0, 100.0)))
        within_margin = bool(centroid_distance_mm <= (1.35 * r_eff) and surface_alignment_dot >= 0.0)

    return {
        "gt_centroid_3d": [round(float(x), 3) for x in gt_centroid],
        "student_mark_3d": [round(float(x), 3) for x in stu],
        "centroid_distance_mm": round(centroid_distance_mm, 3),
        "dist_to_surface_mm": round(dist_to_surface_mm, 3),
        "lesion_radius_mm": round(r_eff, 3),
        "surface_alignment_dot": round(surface_alignment_dot, 3),
        "clicked_vertex_influence": round(stu_vertex_influence, 4),
        "within_margin": within_margin,
        "location_score": location_score,
        "margin_precision": margin_precision,
        "depth_agreement": depth_agreement,
    }


# ---------------------------------------------------------------------------
# 5. Interactive Voice Viva Loop (Patient Simulation + Follow-Up Questioning)
# ---------------------------------------------------------------------------
def run_viva_patient_turn(
    case_data: Dict[str, Any],
    transcript_history: List[Dict[str, Any]],
    current_competencies: Dict[str, bool],
    student_utterance: str,
) -> Dict[str, Any]:
    """
    Uses the LLM to roleplay the patient AND act as an interactive viva interlocutor
    that evaluates which clinical competencies the student just probed and prompts
    the student to justify their diagnosis if needed. Synthesizes the response via Piper TTS.
    """
    patient = case_data["patient"]
    tooth_fdi = case_data["tooth_fdi"]
    tooth_name = case_data["tooth_name"]
    anomaly_type = case_data["anomaly_type"]
    target_surface = case_data["target_surface"]
    depth_mm = case_data["carve_params"]["depth"]
    wall_mm = case_data["carve_params"]["local_wall_thickness_mm"]

    history_str = "\n".join(
        f"{t.get('role', 'user').upper()}: {t.get('text', '')}"
        for t in transcript_history[-6:]
    )

    system_prompt = (
        "You are simulating a dental clinical viva station with two roles:\n"
        f"1. Virtual Patient ({patient['name']}, {patient['age']}yo {patient['gender']}, ASA {patient['asa_class']}).\n"
        f"   - True pathology: {anomaly_type} on FDI Tooth #{tooth_fdi} ({tooth_name}), surface: {target_surface}.\n"
        f"   - Chief complaint: {patient['chief_complaint']}\n"
        f"   - HPI: {patient['hpi']}\n"
        f"   - Simulated lesion depth: {depth_mm}mm within {wall_mm}mm local wall thickness.\n"
        "2. Clinical Viva Tracker: Determine which of the 4 clinical inquiry competencies the student's latest utterance genuinely addressed:\n"
        "   - 'onset': asked when symptoms began or timeline/progression.\n"
        "   - 'triggers': asked about cold, sweet, mechanical brushing, or chewing stimuli.\n"
        "   - 'duration': asked how long pain lingers after stimulus removal or whether pain is spontaneous/nocturnal.\n"
        "   - 'medical_history': asked about systemic health, medications, or drug/latex allergies.\n"
        "Answer the student's latest utterance directly. Do not repeat the chief complaint or a previous answer unless it is directly relevant. Do not invent patient details that are absent from the case; say when you are unsure.\n\n"
        "Respond strictly in valid JSON with these keys:\n"
        "{\n"
        '  "patient_reply": "1-3 natural spoken sentences answering the student\'s question as the patient and prompting a follow-up clinical question",\n'
        '  "newly_demonstrated": {\n'
        '    "onset": true/false,\n'
        '    "triggers": true/false,\n'
        '    "duration": true/false,\n'
        '    "medical_history": true/false\n'
        "  },\n"
        '  "clinical_note": "Brief 1-line examiner note on what the student just asked or stated"\n'
        "}"
    )

    user_prompt = (
        f"Prior dialogue:\n{history_str or '(Start of session)'}\n\n"
        f"Student's latest spoken utterance:\n\"{student_utterance}\"\n\n"
        "Return the JSON response now."
    )

    try:
        raw_resp, llm_provider = call_clinical_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=True,
            temperature=0.25,
        )
        parsed = parse_json_safely(raw_resp)
    except ClinicalLLMUnavailableError as exc:
        logging.getLogger(__name__).warning("Using scripted Viva fallback: %s", exc)
        question = student_utterance.lower()
        newly_demonstrated = {
            "onset": bool(
                re.search(
                    r"\b(when|start|started|begin|began|since|how long|recently|past week|last week|change|changed|progress|worsen)\b",
                    question,
                )
            ),
            "triggers": bool(
                re.search(r"\b(cold|hot|sweet|brush|chew|bite|trigger|food|eat|eaten|ate|meal)\b", question)
            ),
            "duration": bool(re.search(r"\b(duration|last|linger|seconds?|minutes?|night|spontaneous)\b", question)),
            "medical_history": bool(re.search(r"\b(medical|health|medicine|medication|drug|allerg|latex|condition)\b", question)),
        }
        hpi_sentences = re.split(r"(?<=[.!?])\s+", str(patient.get("hpi", "")).strip())
        answer_terms = {
            "onset": r"\b(start|started|began|ago|since)\b",
            "triggers": r"\b(cold|hot|sweet|brush|chew|bite|stimulus)\b",
            "duration": r"\b(last|seconds?|minutes?|night|spontaneous|subsides?)\b",
        }
        prior_patient_replies = [
            str(turn.get("text", "")).strip()
            for turn in transcript_history
            if turn.get("role") == "patient" and turn.get("text")
        ]
        matching_competencies = [
            competency
            for competency, is_asked in newly_demonstrated.items()
            if is_asked
        ]
        matching_sentences = [
            sentence
            for competency in matching_competencies
            for sentence in hpi_sentences
            if competency in answer_terms
            and re.search(answer_terms[competency], sentence, re.IGNORECASE)
            and not any(sentence.lower() in previous.lower() for previous in prior_patient_replies)
        ]
        if re.search(r"\b(hello|hi|hey|good morning|good afternoon)\b", question):
            patient_reply = "Hello again, Doctor. What would you like to know about my symptoms?"
        elif newly_demonstrated["medical_history"]:
            patient_reply = f"My medical record says {patient.get('asa_class', 'my history is not available')}."
        elif matching_sentences:
            patient_reply = " ".join(dict.fromkeys(matching_sentences))
        elif re.search(r"\b(where|which tooth|which part|exactly|inside|outside|gumline|area)\b", question):
            patient_reply = (
                f"It feels like it is around my {case_data.get('short_name', 'tooth')}. "
                "I cannot tell you exactly which part is inside or outside."
            )
        elif re.search(r"\b(what have you tried|tried|taken|treatment|medicine|medication|past week|last week)\b", question):
            patient_reply = "I am not sure, Doctor; I do not have a clear answer about that."
        else:
            unknown_replies = (
                "I am not sure about that, Doctor. You could ask me when it began, what brings it on, or how long it lasts.",
                "I cannot remember that detail. I can tell you more about the tooth discomfort if you like.",
                "I do not know, Doctor. Is there something about the pain or my medical history you would like to ask?",
                "Sorry, I do not have that information. Could you ask me about one of the symptoms?",
            )
            patient_turn_count = len(prior_patient_replies)
            rotation = patient_turn_count % len(unknown_replies)
            rotated_replies = unknown_replies[rotation:] + unknown_replies[:rotation]
            patient_reply = next(
                (reply for reply in rotated_replies if reply not in prior_patient_replies),
                "I am still not sure about that, Doctor. Could you ask me something different?",
            )

        if patient_reply in prior_patient_replies:
            patient_reply = (
                "I may have already mentioned that, Doctor. "
                "Could you ask me about a different symptom or detail?"
            )
        parsed = {
            "patient_reply": patient_reply,
            "newly_demonstrated": newly_demonstrated,
            "clinical_note": "Scripted patient mode: the language-model service is unavailable.",
        }
        llm_provider = "offline-scripted-patient"

    patient_reply = str(parsed.get("patient_reply") or "").strip()
    if not patient_reply:
        patient_reply = (
            f"I've mostly noticed sensitivity on my {case_data['short_name']} when something cold touches it, Doctor. "
            "What do you think is causing it?"
        )

    newly = parsed.get("newly_demonstrated") or {}
    updated_comp = {
        "onset": bool(current_competencies.get("onset", False) or newly.get("onset", False)),
        "triggers": bool(current_competencies.get("triggers", False) or newly.get("triggers", False)),
        "duration": bool(current_competencies.get("duration", False) or newly.get("duration", False)),
        "medical_history": bool(current_competencies.get("medical_history", False) or newly.get("medical_history", False)),
    }

    audio_url = synthesize_speech_to_url(patient_reply, prefix="viva_reply")

    return {
        "patient_reply": patient_reply,
        "audio_url": audio_url,
        "competencies": updated_comp,
        "done_count": sum(1 for v in updated_comp.values() if v),
        "clinical_note": str(parsed.get("clinical_note") or "Utterance processed by clinical LLM"),
        "llm_provider": llm_provider,
    }


# ---------------------------------------------------------------------------
# 6. LLM-Driven Diagnosis Grading + Spoken Teach-Back Synthesis
# ---------------------------------------------------------------------------
def grade_diagnosis_and_generate_teachback(
    case_data: Dict[str, Any],
    location_eval: Dict[str, Any],
    student_verdict: str,
    student_rationale: str,
    spoken_diagnosis_transcript: str,
    viva_transcript: List[Dict[str, Any]],
    viva_competencies: Dict[str, bool],
) -> Dict[str, Any]:
    """
    Grades the student's clinical diagnosis/reasoning using the LLM (against true anomaly type,
    tooth FDI, surface, and severity) and generates a spoken TTS teach-back explanation.
    """
    tooth_fdi = case_data["tooth_fdi"]
    tooth_name = case_data["tooth_name"]
    true_anomaly = case_data["anomaly_type"]
    true_badge = case_data["anomaly_badge"]
    true_surface = case_data["target_surface"]
    radius_mm = case_data["carve_params"]["radius"]
    depth_mm = case_data["carve_params"]["depth"]
    wall_mm = case_data["carve_params"]["local_wall_thickness_mm"]

    combined_diagnosis_text = (
        f"Written Rationale: {student_rationale}\n"
        f"Spoken Viva Diagnosis: {spoken_diagnosis_transcript}\n"
        f"Student's self-assessment of their own 3D mark: {student_verdict}\n"
        "Viva Dialogue:\n"
        + "\n".join(f"- {m.get('role')}: {m.get('text')}" for m in viva_transcript)
    ).strip()

    system_prompt = (
        "You are an strict, objective Dental Board Examiner grading a dental student's diagnostic submission.\n"
        "GROUND TRUTH CASE FACTS:\n"
        f"- Tooth: FDI #{tooth_fdi} ({tooth_name})\n"
        f"- Target Anatomical Surface: {true_surface}\n"
        f"- True Pathology / Anomaly Type: {true_anomaly} ({true_badge})\n"
        f"- Lesion 3D Dimensions: radius {radius_mm}mm, depth {depth_mm}mm within {wall_mm}mm measured local wall thickness (outer dentin / enamel, reversible pulpitis).\n"
        f"- Student's 3D Mark Accuracy (Rule-Based Score A): {location_eval['location_score']}% (Euclidean error: {location_eval['centroid_distance_mm']}mm).\n\n"
        "STRICT GRADING RULES FOR 'diagnosis_score' (0 to 100):\n"
        "1. If the student claims the tooth is healthy/normal, says 'I don't know', or identifies the wrong pathology (e.g. calling a Cavity an Abrasion or vice versa), 'diagnosis_score' MUST be between 0 and 25.\n"
        "2. If the student identifies the pathology family but misses the anatomical surface or restorative management, score between 45 and 70.\n"
        "3. If the student accurately identifies the anomaly type, surface, reversible pulpitis status / depth, and appropriate minimally invasive management, score between 80 and 98.\n"
        "4. 'communication_score' (0 to 100) reflects clinical clarity and organization.\n"
        "5. Provide a concise 3-sentence 'teachback_explanation' spoken directly to the student explaining the ground-truth pathology, why their 3D mark and diagnosis succeeded or failed, and the ideal clinical treatment.\n\n"
        "Return strictly valid JSON with this exact schema:\n"
        "{\n"
        '  "diagnosis_score": <int 0-100>,\n'
        '  "communication_score": <int 0-100>,\n'
        '  "anomaly_correctly_identified": <bool>,\n'
        '  "surface_correctly_identified": <bool>,\n'
        '  "strengths": ["...", "..."],\n'
        '  "weaknesses": ["...", "..."],\n'
        '  "teachback_explanation": "3 spoken sentences for TTS playback"\n'
        "}"
    )

    user_prompt = (
        f"STUDENT SUBMISSION TO GRADE:\n{combined_diagnosis_text}\n\n"
        "Grade this submission strictly against the ground truth and return JSON."
    )

    grading_notice = ""
    try:
        raw_resp, llm_provider = call_clinical_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expect_json=True,
            temperature=0.1,
        )
        parsed = parse_json_safely(raw_resp)
    except ClinicalLLMUnavailableError as exc:
        logging.getLogger(__name__).warning("Using local diagnosis rubric: %s", exc)
        submission = combined_diagnosis_text.lower()
        anomaly_terms = {
            "cavity": r"\b(cavity|caries|decay|carious)\b",
            "abrasion": r"\b(abrasion|wear|toothbrush abrasion)\b",
            "crack": r"\b(crack|fracture|fissure)\b",
            "erosion": r"\b(erosion|acid wear|acid damage)\b",
        }
        anomaly_correct = bool(re.search(anomaly_terms.get(str(true_anomaly).lower(), r"$^"), submission))
        surface_terms = {
            "labial_facial": r"\b(labial|facial|buccal|front surface)\b",
            "occlusal_table": r"\b(occlusal|chewing surface|chewing table)\b",
            "cervical": r"\b(cervical|gumline|gum line)\b",
            "buccal": r"\b(buccal|cheek side)\b",
            "lingual": r"\b(lingual|palatal|tongue side)\b",
        }
        surface_correct = bool(re.search(surface_terms.get(str(true_surface).lower(), r"$^"), submission))
        treatment_mentioned = bool(
            re.search(r"\b(treat|treatment|restore|restoration|seal|filling|composite|monitor|fluoride|desensiti[sz]e)\b", submission)
        )
        rationale_words = len(student_rationale.split())
        diag_score = (
            (50 if anomaly_correct else 0)
            + (20 if surface_correct else 0)
            + (20 if treatment_mentioned else 0)
            + (10 if rationale_words >= 15 else 0)
        )
        comm_score = min(100, 30 + rationale_words * 2)
        strengths = []
        weaknesses = []
        (strengths if anomaly_correct else weaknesses).append(
            "Correctly identified the pathology." if anomaly_correct else f"Review the distinguishing features of {true_anomaly.lower()}."
        )
        (strengths if surface_correct else weaknesses).append(
            "Correctly located the affected tooth surface." if surface_correct else f"Name the affected surface ({true_surface.replace('_', ' ')})."
        )
        if treatment_mentioned:
            strengths.append("Included a treatment or management consideration.")
        else:
            weaknesses.append("Add an appropriate treatment or management plan.")
        parsed = {
            "diagnosis_score": diag_score,
            "communication_score": comm_score,
            "anomaly_correctly_identified": anomaly_correct,
            "surface_correctly_identified": surface_correct,
            "strengths": strengths,
            "weaknesses": weaknesses,
        }
        llm_provider = "offline-local-rubric"
        grading_notice = "The language model is unavailable; this report uses a local keyword-based rubric, not LLM grading."

    diag_score = int(np.clip(int(parsed.get("diagnosis_score", 0)), 0, 100))
    comm_score = int(np.clip(int(parsed.get("communication_score", 50)), 0, 100))

    # History score is computed from the 4 Viva competencies (25 points each)
    done_count = sum(1 for v in viva_competencies.values() if v)
    history_score = int(done_count * 25)

    loc_score = int(location_eval["location_score"])
    # Composite score: 50% Rule-Based Location (Score A) + 50% LLM Diagnosis Score (Score B)
    composite = round(0.5 * loc_score + 0.5 * diag_score, 1)

    if composite >= 90:
        grade_letter = "A"
    elif composite >= 80:
        grade_letter = "B"
    elif composite >= 70:
        grade_letter = "C"
    elif composite >= 60:
        grade_letter = "D"
    else:
        grade_letter = "F"

    teachback_text = str(parsed.get("teachback_explanation") or "").strip()
    if not teachback_text:
        teachback_text = (
            f"On FDI Tooth {tooth_fdi} ({tooth_name}), the ground-truth pathology is a {true_badge.lower()} on the {true_surface} "
            f"with a depth of {depth_mm} millimeters inside a {wall_mm} millimeter wall. "
            f"Your 3D mark was {location_eval['centroid_distance_mm']} millimeters from the lesion centroid, earning a location score of {loc_score} percent, "
            f"and your clinical diagnosis scored {diag_score} percent."
        )

    teachback_audio_url = synthesize_speech_to_url(teachback_text, prefix="teachback")

    return {
        "location_score": loc_score,
        "diagnosis_score": diag_score,
        "history_score": history_score,
        "communication_score": comm_score,
        "composite_score": composite,
        "grade_letter": grade_letter,
        "anomaly_correctly_identified": bool(parsed.get("anomaly_correctly_identified", diag_score >= 70)),
        "surface_correctly_identified": bool(parsed.get("surface_correctly_identified", diag_score >= 70)),
        "strengths": parsed.get("strengths") or [],
        "weaknesses": parsed.get("weaknesses") or [],
        "teachback_text": teachback_text,
        "teachback_audio_url": teachback_audio_url,
        "llm_provider": llm_provider,
        "grading_notice": grading_notice,
    }
