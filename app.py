import os, sys, re, json, torch, numpy as np, pandas as pd, random, logging
from typing import List, Dict, Tuple, Optional, Any
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client
import threading
from surprise import Dataset, Reader, SVD

logger = logging.getLogger()
logger.setLevel(logging.INFO)

for h in logger.handlers[:]:
    logger.removeHandler(h)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)

logger.addHandler(handler)

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

P5_ROOT = os.getenv("P5_ROOT", "/workspace/P5-main")
P5_CKPT = os.getenv("P5_CKPT", "/models/p5/mvt_aug_epoch10.pth")
P5_BACKBONE = os.getenv("P5_BACKBONE", "/models/t5-small")
P5_MAX_LEN = int(os.getenv("P5_MAX_LEN", "256"))
P5_GEN_MAX_LEN = int(os.getenv("P5_GEN_MAX_LEN", "16"))
P5_DROPOUT = float(os.getenv("P5_DROPOUT", "0.1"))
P5_BATCH = int(os.getenv("P5_BATCH", "16"))

# Soft prompt (hf PEFT)
SOFTPROMPT_METHOD = os.getenv("SOFTPROMPT_METHOD", "prompt_tuning")  # 'prompt_tuning' or 'p_tuning'
SOFTPROMPT_VTOKENS = int(os.getenv("SOFTPROMPT_VTOKENS", "40"))
SOFTPROMPT_LR = float(os.getenv("SOFTPROMPT_LR", "5e-4"))
SOFTPROMPT_STEPS = 10
SOFTPROMPT_BSZ = int(os.getenv("SOFTPROMPT_BSZ", "8"))
SOFTPROMPT_INIT_TEXT = os.getenv("SOFTPROMPT_INIT_TEXT", "rating prediction")
HISTORY_MAX_TRAIN = 30

SVD_DIR = os.getenv("SVD_DIR", "/models/svd_surprise")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============= datamaps (movie id mapping for P5) =============
DATAMAPS_PATH = os.getenv("DATAMAPS_PATH", "/models/p5/datamaps.json")
ITEM2ID: Dict[str, str] = {}   # external movie_id(str) -> internal item_id(str)
ID2ITEM: Dict[str, str] = {}   # internal item_id(str) -> external movie_id(str)

def load_datamaps():
    global ITEM2ID, ID2ITEM
    try:
        logger.info(f"Loading datamaps from {DATAMAPS_PATH}")
        with open(DATAMAPS_PATH, "r") as f:
            dm = json.load(f)
        ITEM2ID = dm.get("item2id", {})
        ID2ITEM = {v: k for k, v in ITEM2ID.items()}
        logger.info(f"Datamaps loaded successfully: item2id size={len(ITEM2ID)}")
    except Exception as e:
        logger.error(f"Failed to load datamaps: {e}")
        raise

def map_movie_for_p5(ext_mid: str) -> Optional[str]:
    return ITEM2ID.get(str(ext_mid))

def map_history_for_p5(history: List[Dict], max_n: int = HISTORY_MAX_TRAIN) -> List[Dict]:
    out = []
    for h in history[:max_n]:
        ext = str(h.get("movie_id"))
        internal = map_movie_for_p5(ext)
        if internal is None:
            continue
        out.append({"movie_id": internal, "rating": float(h.get("rating", 0.0))})
    return out

# ===================== Client / App =====================
sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================= SVD =======================
random.seed(2026)
np.random.seed(2026)

SVD_5CORE_PATH = os.getenv("SVD_5CORE_PATH", f"{SVD_DIR}/ratings_5core.csv")
SVD_HISTORY_PATH = os.getenv("SVD_HISTORY_PATH", f"{SVD_DIR}/lab_ratings.csv")
SVD_HP_PATH = os.getenv("SVD_HP_PATH", f"{SVD_DIR}/svd_hyperparams.json")

SVD_RATING_SCALE = (0, 10)

SURPRISE_BASE_RATINGS: Optional[pd.DataFrame] = None
SURPRISE_HP: Dict[str, Any] = {}

def _load_json_if_exists(path: str) -> Dict[str, Any]:
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        print("!!! Missing svd_hyperparams.json !!!")
        return {}

def load_surprise_hparams() -> Dict[str, Any]:
    hp = _load_json_if_exists(SVD_HP_PATH)
    hp = {k: v for k, v in hp.items()}
    return hp

def load_surprise_base_ratings() -> pd.DataFrame:
    if not os.path.exists(SVD_5CORE_PATH):
        raise FileNotFoundError(f"ratings_5core.csv not found: {SVD_5CORE_PATH}")
    if not os.path.exists(SVD_HISTORY_PATH):
        raise FileNotFoundError(f"lab_ratings.csv not found: {SVD_HISTORY_PATH}")

    df5 = pd.read_csv(SVD_5CORE_PATH, dtype={"MovieID": str})
    df5 = df5.rename(columns={"UserID": "user", "MovieID": "item", "Rating": "rating"})
    df5 = df5[["user", "item", "rating"]]

    dfh = pd.read_csv(SVD_HISTORY_PATH, dtype={"item_id": str})
    dfh = dfh.rename(columns={"user_id": "user", "item_id": "item"})
    dfh = dfh[["user", "item", "rating"]]

    df = pd.concat([df5, dfh], ignore_index=True)

    df["user"] = df["user"].astype(str)
    df["item"] = df["item"].astype(str)
    df["rating"] = df["rating"].astype(float)

    df = df.dropna(subset=["user", "item", "rating"])

    return df

def load_surprise_assets():
    global SURPRISE_BASE_RATINGS, SURPRISE_HP
    logger.info(f"Loading Surprise base ratings from: {SVD_5CORE_PATH} + {SVD_HISTORY_PATH}")
    SURPRISE_BASE_RATINGS = load_surprise_base_ratings()
    logger.info(f"Surprise base ratings loaded: rows={len(SURPRISE_BASE_RATINGS)}")

    SURPRISE_HP = load_surprise_hparams()
    logger.info(f"Surprise SVD hyperparams loaded: keys={list(SURPRISE_HP.keys())}")

def _build_user_df(session_id: str, history: List[Dict]) -> pd.DataFrame:
    rows = []
    for h in history:
        mid = str(h.get("movie_id"))
        r = float(h.get("rating", 0.0))
        rows.append({"user": str(session_id), "item": mid, "rating": r})

    dfu = pd.DataFrame(rows)

    dfu["user"] = dfu["user"].astype(str)
    dfu["item"] = dfu["item"].astype(str)
    dfu["rating"] = dfu["rating"].astype(float)

    return dfu

def score_candidates_svd_surprise(session_id: str, history: List[Dict], cand_ids: List[str]) -> List[Tuple[str, float]]:

    if SURPRISE_BASE_RATINGS is None:
        raise RuntimeError("SURPRISE_BASE_RATINGS is not loaded. Call load_surprise_assets() at startup.")

    dfu = _build_user_df(session_id, history)
    if dfu.empty:
        logger.warning("No user ratings available for Surprise SVD scoring")
        return []

    df_all = pd.concat([SURPRISE_BASE_RATINGS, dfu], ignore_index=True)

    reader = Reader(rating_scale=SVD_RATING_SCALE)
    data = Dataset.load_from_df(df_all[["user", "item", "rating"]], reader)
    trainset = data.build_full_trainset()
    
    logger.info(f"Fitting SVD (params - {SURPRISE_HP})")
    algo = SVD(**(SURPRISE_HP or {}))
    algo.fit(trainset)

    out: List[Tuple[str, float]] = []
    uid = str(session_id)

    for cid in cand_ids:
        iid = str(cid)
        est = algo.predict(uid, iid).est
        out.append((iid, float(est)))

    logger.info(f"Surprise SVD scored {len(out)} candidates")
    return out



# ===================== P5 =========================
sys.path.extend([P5_ROOT, os.path.join(P5_ROOT, "src")])
from transformers import T5Config
from src.tokenization import P5Tokenizer
from src.pretrain_model import P5Pretraining
from src.utils import load_state_dict

from peft import get_peft_model, PromptTuningConfig, PromptEncoderConfig, TaskType

TOKENIZER, BASE_STATE = None, None

def create_config_eval():
    # Load config from local cache only (no network calls)
    cfg = T5Config.from_pretrained(P5_BACKBONE, local_files_only=True)
    cfg.dropout_rate = P5_DROPOUT
    cfg.dropout = P5_DROPOUT
    cfg.attention_dropout = P5_DROPOUT
    cfg.activation_dropout = P5_DROPOUT
    cfg.losses = "rating"
    return cfg

def load_tokenizer_once():
    global TOKENIZER
    if TOKENIZER is not None:
        return
    try:
        logger.info(f"Loading P5 tokenizer from {P5_BACKBONE}")
        # Load tokenizer from local cache only (no network calls)
        TOKENIZER = P5Tokenizer.from_pretrained(
            P5_BACKBONE,
            max_length=P5_MAX_LEN,
            do_lower_case=False,
            local_files_only=True
        )
        logger.info("P5 tokenizer loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load P5 tokenizer: {e}")
        raise

def load_base_state_once():
    global BASE_STATE
    if BASE_STATE is not None:
        return
    try:
        if os.path.exists(P5_CKPT):
            logger.info(f"Loading P5 checkpoint from {P5_CKPT}")
            BASE_STATE = load_state_dict(P5_CKPT, DEVICE)
            logger.info("P5 checkpoint loaded successfully")
        else:
            logger.warning(f"P5 checkpoint not found at {P5_CKPT}")
            BASE_STATE = None
    except Exception as e:
        logger.error(f"Failed to load P5 checkpoint: {e}")
        raise

def create_per_request_base():
    cfg = create_config_eval()
    model = P5Pretraining.from_pretrained(P5_BACKBONE, config=cfg, local_files_only=True).to(DEVICE)
    model.resize_token_embeddings(TOKENIZER.vocab_size)
    model.tokenizer = TOKENIZER
    model.eval()
    if BASE_STATE is not None:
        _ = model.load_state_dict(BASE_STATE, strict=False)
    return model

def attach_soft_prompt(model, tokenizer,
                       method: str = SOFTPROMPT_METHOD,
                       num_virtual_tokens: int = SOFTPROMPT_VTOKENS,
                       init_text: Optional[str] = SOFTPROMPT_INIT_TEXT):
    if method not in {"prompt_tuning", "p_tuning"}:
        raise ValueError("SOFTPROMPT_METHOD must be 'prompt_tuning' or 'p_tuning'")

    # Move model to CPU temporarily to avoid device mismatch during PEFT initialization
    device = next(model.parameters()).device
    model = model.cpu()

    if method == "prompt_tuning":
        cfg = PromptTuningConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=num_virtual_tokens,
            tokenizer_name_or_path=getattr(tokenizer, "name_or_path", P5_BACKBONE),
            prompt_tuning_init="TEXT" if init_text else "RANDOM",
            prompt_tuning_init_text=init_text or "rating prediction",
        )
    else:  # p_tuning v2
        cfg = PromptEncoderConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            num_virtual_tokens=num_virtual_tokens,
            encoder_hidden_size=128,
        )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()

    # Move back to original device
    peft_model = peft_model.to(device)
    return peft_model

def _first_float(text: str, default: float = -1.0) -> float:
    m = re.search(r"-?\d+(\.\d+)?", text.strip())
    return float(m.group(0)) if m else default

def make_p5_prompt(session_id: str, movie_id: str, history: Optional[List[Dict]] = None) -> str:
    base_prompt = f"Which star rating will user_{session_id} give movie_{movie_id}?"
    if history and len(history) > 0:
        history_str = "Previous ratings: "
        for h in history[-30:]:
            hist_mid = str(h.get("movie_id", "unknown"))
            hist_rating = float(h.get("rating", 0.0)) 
            # Use integer format for display since user ratings are integers
            history_str += f"movie_{hist_mid}:{int(hist_rating)}, "
        history_str = history_str.rstrip(", ")
        return f"{history_str}. {base_prompt} (0.0 being lowest and 10.0 being highest, decimals allowed)"
    return f"{base_prompt} (0.0 being lowest and 10.0 being highest, decimals allowed)"

def _build_training_examples(session_id: str, history: List[Dict]) -> List[Tuple[str, str]]:
    """
    returns list of (src_prompt, tgt_text) where tgt_text is like '4.0'
    For each known rating, we create the training example with exact integer rating
    """
    exs = []
    for h in history[:HISTORY_MAX_TRAIN]:
        mid = str(h["movie_id"])
        rating = float(h["rating"])
        src = make_p5_prompt(session_id, mid, history)
        # Keep integer format for known ratings
        tgt = f"{int(rating)}"
        exs.append((src, tgt))
    return exs

def finetune_soft_prompt(per_user_model, tokenizer, session_id: str, history: List[Dict],
                         lr: float = SOFTPROMPT_LR, steps: int = SOFTPROMPT_STEPS, bsz: int = SOFTPROMPT_BSZ):
    exs = _build_training_examples(session_id, history)
    if not exs:
        logger.warning(f"No training examples available for session {session_id}")
        return

    logger.info(f"Starting soft prompt finetuning for session {session_id} with {len(exs)} examples")
    device = next(per_user_model.parameters()).device
    per_user_model.train()
    optim = torch.optim.AdamW([p for p in per_user_model.parameters() if p.requires_grad], lr=lr)

    random.shuffle(exs)
    step = 0
    i = 0
    while step < steps:
        batch = exs[i:i+bsz]
        if not batch:
            i = 0
            continue
        i += bsz
        srcs = [s for s, _ in batch]
        tgts = [t for _, t in batch]

        # AMP can be further added if needed.
        enc = tokenizer(srcs, return_tensors="pt", padding=True, truncation=True, max_length=P5_MAX_LEN).to(device)
        with tokenizer.as_target_tokenizer():
            dec = tokenizer(tgts, return_tensors="pt", padding=True, truncation=True, max_length=8).to(device)
        labels = dec["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        optim.zero_grad(set_to_none=True)
        out = per_user_model(**enc, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(per_user_model.parameters(), 1.0)
        optim.step()

        if (step + 1) % 10 == 0:
            logger.info(f"Soft prompt finetuning step {step+1}/{steps} loss={loss.item():.4f}")
        step += 1

    per_user_model.eval()
    logger.info(f"Soft prompt finetuning completed for session {session_id}")

@torch.no_grad()
def p5_score_candidates_mapped(model, tokenizer, session_id: str,
                               history_mapped: List[Dict], mapped_ids: List[str]) -> List[float]:
    res: List[float] = []
    texts = [make_p5_prompt(session_id, mid, history_mapped) for mid in mapped_ids]
    logger.info(f"P5 scoring {len(texts)} candidates for session {session_id}")

    for s in range(0, len(texts), P5_BATCH):
        batch = texts[s:s+P5_BATCH]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=P5_MAX_LEN)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        # Log first batch for debugging
        if s == 0 and len(batch) > 0:
            logger.info(f"Sample prompt: {batch[0][:200]}...")
            logger.info(f"Input tokens: {enc['input_ids'].shape}")

        out = model.generate(**enc, max_new_tokens=P5_GEN_MAX_LEN, num_beams=2, do_sample=False)
        
        dec = tokenizer.batch_decode(out, skip_special_tokens=True)

        # Log first batch outputs for debugging
        if s == 0 and len(dec) > 0:
            logger.info(f"Sample generated outputs (first 3): {dec[:3]}")

        for txt in dec:
            score = _first_float(txt, default=-1.0)
            res.append(score)

    logger.info(f"P5 scoring completed, generated {len(res)} scores")
    logger.info(f"Score distribution: min={min(res) if res else 'N/A'}, max={max(res) if res else 'N/A'}, mean={sum(res)/len(res) if res else 'N/A'}")
    return res

# ===================== Supabase I/O =====================
def get_history(session_id: str, limit: int = 50) -> List[Dict]:
    try:
        logger.info(f"Fetching history for session {session_id}")
        r = sb.table("movie_ratings")\
            .select("movie_id,rating")\
            .eq("session_id", session_id)\
            .order("created_at", desc=True)\
            .limit(limit).execute()
        history = r.data or []
        logger.info(f"Retrieved {len(history)} ratings for session {session_id}")
        return history
    except Exception as e:
        logger.error(f"Failed to fetch history for session {session_id}: {e}")
        return []

def get_phase1_movies(session_id: str) -> List[str]:
    """
    Get the list of Phase 1 movies for a given session.
    These movies should be excluded from Phase 2 recommendations.
    """
    try:
        logger.info(f"Fetching Phase 1 movies for session {session_id}")
        r = sb.table("session_phase1_movies").select("movie_id").eq("session_id", session_id).execute()
        phase1_movies = [str(row["movie_id"]) for row in (r.data or [])]
        logger.info(f"Retrieved {len(phase1_movies)} Phase 1 movies to exclude")
        return phase1_movies
    except Exception as e:
        logger.error(f"Failed to fetch Phase 1 movies: {e}")
        return []

def get_candidates(limit: int = 10000, exclude_movies: List[str] = None) -> List[str]:
    """
    Get candidate movies from phase2_movies, optionally excluding specific movies.

    Args:
        limit: Maximum number of candidates to fetch
        exclude_movies: List of movie IDs to exclude (e.g., Phase 1 movies)
    """
    try:
        logger.info("Fetching candidate movies from phase2_movies")
        r = sb.table("phase2_movies").select("id").limit(limit).execute()
        candidates = [str(row["id"]) for row in (r.data or [])]

        if exclude_movies:
            original_count = len(candidates)
            candidates = [c for c in candidates if c not in exclude_movies]
            excluded_count = original_count - len(candidates)
            logger.info(f"Excluded {excluded_count} Phase 1 movies from candidates")

        logger.info(f"Retrieved {len(candidates)} candidate movies")
        return candidates
    except Exception as e:
        logger.error(f"Failed to fetch candidates: {e}")
        return []

# ====== display_order ======
def build_display_sequence(p5_top: List[Tuple[str, float]],
                           svd_top: List[Tuple[str, float]]) -> List[Tuple[str, str, int]]:
    """
    Interleaved Displaying Order: starting model is random.
    - Take all 10 from each model as candidates.
    - Alternate between models: Model1 #1, Model2 #1, Model1 #2, Model2 #2, etc.
    - If a movie is duplicated, skip it in the later model and move to the next rank.
    - Continue until we have 10 unique movies.
    Return: [(model, movie_id, display_order 1..10)]
    """
    p5_list = p5_top[:10]  # Use all 10 ranks
    svd_list = svd_top[:10]  # Use all 10 ranks
    i_p5 = 0
    i_svd = 0
    need_p5 = 10  # Try to get up to 10 from P5
    need_svd = 10  # Try to get up to 10 from SVD
    used = set()
    seq: List[Tuple[str, str, int]] = []

    # Decide the starting model randomly
    turn = random.choice(["p5", "svd"])
    logger.info(f"Building display sequence starting with {turn}")

    # Continue until we have 10 unique movies or run out of candidates
    while len(seq) < 10 and (i_p5 < len(p5_list) or i_svd < len(svd_list)):
        if turn == "p5":
            # Skip duplicates in P5
            while i_p5 < len(p5_list) and p5_list[i_p5][0] in used:
                i_p5 += 1
            if i_p5 < len(p5_list):
                mid = p5_list[i_p5][0]
                seq.append(("p5", mid, len(seq)+1))
                used.add(mid)
                i_p5 += 1
            turn = "svd"
        elif turn == "svd":
            # Skip duplicates in SVD
            while i_svd < len(svd_list) and svd_list[i_svd][0] in used:
                i_svd += 1
            if i_svd < len(svd_list):
                mid = svd_list[i_svd][0]
                seq.append(("svd", mid, len(seq)+1))
                used.add(mid)
                i_svd += 1
            turn = "p5"

        # Safety check: if both models are exhausted, break
        if i_p5 >= len(p5_list) and i_svd >= len(svd_list):
            break

    logger.info(f"Display sequence built with {len(seq)} movies (P5 used: {sum(1 for m, _, _ in seq if m == 'p5')}, SVD used: {sum(1 for m, _, _ in seq if m == 'svd')})")
    return seq

def rows_from_scored(session_id: str, model: str, scored: List[Tuple[str, float]],
                     topk: int, phase: int) -> List[Dict]:
    """
    Store the model's internal TopK (=10) as rank 1..TopK.
    Set display_order to None as default; fill it later when applying the display order.
    """
    top = sorted(scored, key=lambda x: x[1], reverse=True)[:topk]
    rows = []
    for i, (mid, sc) in enumerate(top):
        rows.append({
            "session_id": session_id,
            "movie_id": mid,
            "score": float(sc),
            "model": model,
            "phase": phase,
            "rank": i+1,
            "display_order": None
        })
    logger.info(f"Created {len(rows)} recommendation rows for model {model}")
    return rows

def upsert_rows(rows: List[Dict]):
    if rows:
        try:
            logger.info(f"Upserting {len(rows)} recommendation rows to database")
            # Delete existing recommendations for this session/model/phase before inserting new ones
            if rows:
                session_id = rows[0]["session_id"]
                model = rows[0]["model"]
                phase = rows[0]["phase"]
                sb.table("recommendations").delete().eq("session_id", session_id).eq("model", model).eq("phase", phase).execute()
                logger.info(f"Deleted existing {model} recommendations for session {session_id}")

            # Now insert the new rows
            sb.table("recommendations").insert(rows).execute()
            logger.info("Recommendation rows inserted successfully")
        except Exception as e:
            logger.error(f"Failed to upsert recommendation rows: {e}")
            raise

# ===================== API =====================
class RecReq(BaseModel):
    session_id: str
    topk_per_model: int = 10
    phase: int = 2

# Global initialization state
READY = {"ok": False, "msg": "booting"}

def _heavy_init():
    """Initialize all heavy models in background thread"""
    try:
        logger.info("Starting heavy initialization...")

        # ENV Validity check
        required = ["SUPABASE_URL", "SUPABASE_SERVICE_KEY"]
        missing = [k for k in required if not os.getenv(k)]
        if missing:
            raise RuntimeError(f"Missing env: {', '.join(missing)}")
        logger.info("Environment variables validated")

        # Mark as ready FIRST with limited functionality
        # This prevents Cloud Run from killing the instance
        READY.update(ok=True, msg="ready_basic")
        logger.info("Basic initialization complete, service is ready")

        logger.info("Loading Surprise SVD assets (critical)...")
        load_surprise_assets()
        logger.info("Surprise SVD assets loaded")

        logger.info("Loading datamaps (critical)...")
        load_datamaps()
        logger.info("Datamaps loaded")

        logger.info("Loading P5 tokenizer...")
        load_tokenizer_once()
        logger.info("P5 tokenizer loaded")

        logger.info("Loading P5 base state...")
        load_base_state_once()
        logger.info("P5 base state loaded")

        READY.update(ok=True, msg="ready_full")
        logger.info("Full initialization completed successfully")

    except Exception as e:
        error_msg = f"init_error: {e}"
        READY.update(ok=False, msg=error_msg)
        logger.error(f"Heavy initialization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

@app.on_event("startup")
def _startup():
    logger.info("FastAPI startup event triggered")
    logger.info(f"Starting background initialization thread")
    threading.Thread(target=_heavy_init, daemon=True).start()

@app.get("/health")
def health():
    logger.info(f"Health check requested - Status: {READY['msg']}")
    return {
        "ok": READY["ok"],
        "status": READY["msg"],
        "device": DEVICE,
        "p5_ckpt": P5_CKPT,
        "svd_dir": SVD_DIR,
        "n_items": int(SURPRISE_BASE_RATINGS["item"].nunique() if SURPRISE_BASE_RATINGS is not None else 0)
    }

@app.post("/recommend")
def recommend(req: RecReq, x_webhook_secret: Optional[str] = Header(None)):
    logger.info(f"Recommendation request received for session {req.session_id}")
    logger.info(f"Service status: {READY['msg']}")

    if not READY["ok"]:
        logger.warning(f"Service not ready: {READY['msg']}")
        raise HTTPException(503, "warming up")

    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        logger.warning("Invalid webhook secret provided")
        raise HTTPException(401, "bad secret")

    # Check if we have full P5 capabilities
    use_p5 = (READY["msg"] == "ready_full" and TOKENIZER is not None and BASE_STATE is not None)
    if not use_p5:
        logger.warning("P5 model not fully loaded, will use SVD-only recommendations")

    try:
        # 1) Input acquisition
        logger.info("Step 1: Acquiring input data")
        hist = get_history(req.session_id)
        if not hist:
            logger.error(f"No rating history found for session {req.session_id}")
            raise HTTPException(400, "No rating history for the given session_id.")

        # Get Phase 1 movies to exclude from recommendations
        phase1_movies = get_phase1_movies(req.session_id)
        logger.info(f"Excluding {len(phase1_movies)} Phase 1 movies from recommendations")

        all_candidates = get_candidates(exclude_movies=phase1_movies)
        if not all_candidates:
            logger.error("No candidates found in phase2_movies after exclusions")
            raise HTTPException(400, "No candidates found for phase2_movies after excluding Phase 1 movies.")

        # 2) SVD Top-100 Scores
        logger.info("Step 2: Computing SVD recommendations")
        svd_scored_all = score_candidates_svd_surprise(req.session_id, hist, all_candidates)
        svd_top100 = sorted(svd_scored_all, key=lambda x: x[1], reverse=True)[:100]
        svd_rows = rows_from_scored(req.session_id, "svd", svd_top100, topk=req.topk_per_model, phase=req.phase)
        logger.info(f"SVD generated {len(svd_top100)} scored candidates")

        # 3) P5 soft prompt: create per-user base + attach adapter + short finetuning
        p5_rows = []
        if use_p5:
            try:
                logger.info("Step 3: Preparing P5 model")
                hist_mapped = map_history_for_p5(hist, max_n=HISTORY_MAX_TRAIN)
                logger.info(f"Mapped {len(hist_mapped)} history items for P5")

                base = create_per_request_base()
                base.eval()

                # 4) P5: Rerank only SVD Top-100 (Using trained per_user model)
                logger.info("Step 4: P5 reranking of SVD top candidates")
                pairs = []  # [(ext_mid, internal_id)]
                for ext_mid, _ in svd_top100:
                    internal = map_movie_for_p5(ext_mid)
                    if internal is not None:
                        pairs.append((ext_mid, internal))

                if pairs:
                    mapped_ids = [internal for _, internal in pairs]
                    p5_scores = p5_score_candidates_mapped(base, TOKENIZER, req.session_id, hist_mapped, mapped_ids)
                    # Pair score with external movie_id
                    p5_scored_on_100 = [(ext_mid, float(sc)) for (ext_mid, _), sc in zip(pairs, p5_scores)]
                else:
                    logger.warning("No movies could be mapped for P5 scoring")
                    p5_scored_on_100 = []

                p5_rows = rows_from_scored(req.session_id, "p5", p5_scored_on_100, topk=req.topk_per_model, phase=req.phase)
            except Exception as e:
                logger.error(f"P5 processing failed, continuing with SVD only: {e}")
                import traceback
                logger.error(traceback.format_exc())
                p5_rows = []
        else:
            logger.info("Step 3-4: Skipping P5 processing (model not available)")

        # 5) Display order
        logger.info("Step 5: Building display sequence")
        if p5_rows:
            # Pass all 10 movies from each model to build_display_sequence
            p5_top_pairs = [(r["movie_id"], r["score"]) for r in sorted(p5_rows, key=lambda x:x["rank"])][:10]
            svd_top_pairs = [(r["movie_id"], r["score"]) for r in sorted(svd_rows, key=lambda x:x["rank"])][:10]
            display_seq = build_display_sequence(p5_top_pairs, svd_top_pairs)
        else:
            # SVD-only: just use top 10 SVD results
            svd_top_pairs = [(r["movie_id"], r["score"]) for r in sorted(svd_rows, key=lambda x:x["rank"])][:10]
            display_seq = [("svd", mid, i+1) for i, (mid, _) in enumerate(svd_top_pairs)]

        # 6) Reflect display order
        logger.info("Step 6: Applying display order")
        disp_map = {(m, mid): order for (m, mid, order) in display_seq}

        # Apply display_order to the movies in the display sequence
        if p5_rows:
            for r in p5_rows:
                key = ("p5", r["movie_id"])
                if key in disp_map:
                    r["display_order"] = disp_map[key]
        for r in svd_rows:
            key = ("svd", r["movie_id"])
            if key in disp_map:
                r["display_order"] = disp_map[key]

        # Count how many movies got display_order
        movies_with_display_order = sum(1 for r in (p5_rows + svd_rows) if r.get("display_order") is not None)
        logger.info(f"Assigned display_order to {movies_with_display_order} movies from the interleaved sequence")

        # 7) upsert
        logger.info("Step 7: Saving recommendations to database")
        if p5_rows:
            upsert_rows(p5_rows)
        upsert_rows(svd_rows)

        result = {
            "session_id": req.session_id,
            "phase": req.phase,
            "svd_top_saved": len(svd_rows),    # 10
            "p5_top_saved": len(p5_rows),      # 10
            "svd_top100_size": len(svd_top100),
            "display_sequence": display_seq
        }
        
        logger.info(f"Recommendation generation completed successfully for session {req.session_id}")
        logger.info(f"Result: {result}")
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during recommendation generation: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")
