import os, sys, re, json, torch, numpy as np, random
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from supabase import create_client, Client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

P5_ROOT = os.getenv("P5_ROOT", "/workspace/P5-main")
P5_CKPT = os.getenv("P5_CKPT", "/models/p5/checkpoint.pth")
P5_BACKBONE = os.getenv("P5_BACKBONE", "t5-small")
P5_MAX_LEN = int(os.getenv("P5_MAX_LEN", "256"))
P5_GEN_MAX_LEN = int(os.getenv("P5_GEN_MAX_LEN", "6"))
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

SVD_DIR = os.getenv("SVD_DIR", "/models/svd/v1")  # Directory where SVD files are stored (e.g., V.npy, ...)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============= datamaps (movie id mapping for P5) =============
DATAMAPS_PATH = os.getenv("DATAMAPS_PATH", "/models/p5/datamaps.json")
ITEM2ID: Dict[str, str] = {}   # external movie_id(str) -> internal item_id(str)
ID2ITEM: Dict[str, str] = {}   # internal item_id(str) -> external movie_id(str)

def load_datamaps():
    global ITEM2ID, ID2ITEM
    with open(DATAMAPS_PATH, "r") as f:
        dm = json.load(f)
    ITEM2ID = dm.get("item2id", {})
    ID2ITEM = {v: k for k, v in ITEM2ID.items()}
    print(f"[datamaps] loaded: item2id size={len(ITEM2ID)} from {DATAMAPS_PATH}")

def map_movie_for_p5(ext_mid: str) -> Optional[str]:
    return ITEM2ID.get(str(ext_mid))

def map_history_for_p5(history: List[Dict], max_n: int = HISTORY_MAX_TRAIN) -> List[Dict]:
    out = []
    for h in history[:max_n]:
        ext = str(h.get("movie_id"))
        internal = map_movie_for_p5(ext)
        if internal is None:
            continue
        out.append({"movie_id": internal, "ratings": float(h.get("ratings", 0.0))})
    return out

# ===================== Client / App =====================
sb: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
app = FastAPI()

# ======================= SVD =======================
V: Optional[np.ndarray] = None
ITEM_IDS: Optional[np.ndarray] = None
ITEM_BIAS: Optional[np.ndarray] = None
MU: float = 0.0
REG: float = 0.05
IDX: Dict[str, int] = {}

def load_svd():
    global V, ITEM_IDS, ITEM_BIAS, MU, REG, IDX
    V = np.load(f"{SVD_DIR}/V.npy").astype("float32")
    ITEM_IDS = np.array(json.load(open(f"{SVD_DIR}/item_ids.json")))
    ITEM_BIAS = np.load(f"{SVD_DIR}/item_bias.npy").astype("float32") if os.path.exists(f"{SVD_DIR}/item_bias.npy") \
                else np.zeros(len(ITEM_IDS), dtype="float32")
    MU = json.load(open(f"{SVD_DIR}/global_mean.json")).get("mu", 7.31) if os.path.exists(f"{SVD_DIR}/global_mean.json") else 7.31
    REG = json.load(open(f"{SVD_DIR}/lambda.json")).get("reg", 0.05) if os.path.exists(f"{SVD_DIR}/lambda.json") else 0.05
    IDX = {str(mid): i for i, mid in enumerate(ITEM_IDS)}

def infer_user_vec_svd(history: List[Dict]) -> Optional[np.ndarray]:
    idxs, y = [], []
    for h in history:
        mid = h.get("movie_id")
        r = float(h.get("ratings", 0))
        ix = IDX.get(mid)
        if ix is None: 
            continue
        idxs.append(ix)
        y.append(r - MU - ITEM_BIAS[ix])
    if not idxs:
        return None
    V_R = V[idxs, :]
    y = np.array(y, dtype="float32")
    A = V_R.T @ V_R
    A[np.diag_indices_from(A)] += REG
    p_u = np.linalg.solve(A, V_R.T @ y)
    return p_u.astype("float32")

def score_candidates_svd(p_u: np.ndarray, cand_ids: List[str]) -> List[Tuple[str, float]]:
    pairs = [(cid, IDX.get(cid)) for cid in cand_ids]
    pairs = [(cid, ix) for cid, ix in pairs if ix is not None]
    if not pairs:
        return []
    ix = np.array([ix for _, ix in pairs], dtype=int)
    scores = (V[ix, :] @ p_u).astype("float32")
    return [(cid, float(s)) for (cid, _), s in zip(pairs, scores)]

# ===================== P5 (Similar with test_ml1m_small.py) =====================
sys.path.extend([P5_ROOT, os.path.join(P5_ROOT, "src")])
from transformers import T5Config
from src.tokenization import P5Tokenizer
from src.pretrain_model import P5Pretraining
from src.utils import load_state_dict

from peft import get_peft_model, PromptTuningConfig, PromptEncoderConfig, TaskType

TOKENIZER, BASE_STATE = None, None  # 전역 캐시(디스크 I/O 최소화)

def create_config_eval():
    cfg = T5Config.from_pretrained(P5_BACKBONE)
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
    TOKENIZER = P5Tokenizer.from_pretrained(P5_BACKBONE, max_length=P5_MAX_LEN, do_lower_case=False)

def load_base_state_once():
    global BASE_STATE
    if BASE_STATE is not None:
        return
    if os.path.exists(P5_CKPT):
        BASE_STATE = load_state_dict(P5_CKPT, DEVICE)
        print(f"[P5] checkpoint state cached from {P5_CKPT}")
    else:
        BASE_STATE = None
        print(f"[P5] WARNING: checkpoint not found at {P5_CKPT}")

def create_per_request_base():
    cfg = create_config_eval()
    model = P5Pretraining.from_pretrained(P5_BACKBONE, config=cfg).to(DEVICE)
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
    return peft_model

def _first_float(text: str, default: float = -1.0) -> float:
    m = re.search(r"-?\d+(\.\d+)?", text.strip())
    return float(m.group(0)) if m else default

def make_p5_prompt(session_id: str, movie_id: str, history: Optional[List[Dict]] = None) -> str:
    return f"Which star rating will user_{session_id} give movie_{movie_id}? (0 being lowest and 10 being highest)"

def _build_training_examples(session_id: str, history: List[Dict]) -> List[Tuple[str, str]]:
    """
    returns list of (src_prompt, tgt_text) where tgt_text is like '4.0'
    """
    exs = []
    for h in history[:HISTORY_MAX_TRAIN]:
        mid = str(h["movie_id"])
        rating = float(h["ratings"])
        src = make_p5_prompt(session_id, mid, history)
        tgt = f"{rating:.1f}"
        exs.append((src, tgt))
    return exs

def finetune_soft_prompt(per_user_model, tokenizer, session_id: str, history: List[Dict],
                         lr: float = SOFTPROMPT_LR, steps: int = SOFTPROMPT_STEPS, bsz: int = SOFTPROMPT_BSZ):
    exs = _build_training_examples(session_id, history)
    if not exs:
        return

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
            print(f"[softprompt] step {step+1}/{steps} loss={loss.item():.4f}")
        step += 1

    per_user_model.eval()

@torch.no_grad()
def p5_score_candidates_mapped(model, tokenizer, session_id: str,
                               history_mapped: List[Dict], mapped_ids: List[str]) -> List[float]:
    res: List[float] = []
    texts = [make_p5_prompt(session_id, mid, history_mapped) for mid in mapped_ids]
    for s in range(0, len(texts), P5_BATCH):
        batch = texts[s:s+P5_BATCH]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=P5_MAX_LEN)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        out = model.generate(**enc, max_length=P5_GEN_MAX_LEN, num_beams=1)
        dec = tokenizer.batch_decode(out, skip_special_tokens=True)
        for txt in dec:
            res.append(_first_float(txt, default=-1.0))
    return res

# ===================== Supabase I/O =====================
def get_history(session_id: str, limit: int = 50) -> List[Dict]:
    r = sb.table("movie_ratings")\
        .select("movie_id,ratings")\
        .eq("session_id", session_id)\
        .order("created_at", desc=True)\
        .limit(limit).execute()
    return r.data or []

def get_candidates(limit: int = 10000) -> List[str]:
    r = sb.table("phase2_movies").select("id").limit(limit).execute()
    return [str(row["id"]) for row in (r.data or [])]

# ====== display_order ======
def build_display_sequence(p5_top: List[Tuple[str, float]],
                           svd_top: List[Tuple[str, float]]) -> List[Tuple[str, str, int]]:
    """
    Interleaved Displaying Order: starting model is random.
    - Take the top 5 from each model as candidates.
    - If a movie is duplicated, keep the one from the earlier model and skip it in the later model, moving to the next rank.
    Return: [(model, movie_id, display_order 1..N)]
    """
    p5_list = p5_top[:5]
    svd_list = svd_top[:5]
    i_p5 = 0
    i_svd = 0
    need_p5 = 5
    need_svd = 5
    used = set()
    seq: List[Tuple[str, str, int]] = []

    # Decide the starting model randomly
    turn = random.choice(["p5", "svd"])

    while (need_p5 > 0 or need_svd > 0) and (i_p5 < len(p5_list) or i_svd < len(svd_list)):
        if turn == "p5" and need_p5 > 0:
            while i_p5 < len(p5_list) and p5_list[i_p5][0] in used:
                i_p5 += 1
            if i_p5 < len(p5_list):
                mid = p5_list[i_p5][0]
                seq.append(("p5", mid, len(seq)+1))
                used.add(mid)
                need_p5 -= 1
                i_p5 += 1
            turn = "svd"
        elif turn == "svd" and need_svd > 0:
            while i_svd < len(svd_list) and svd_list[i_svd][0] in used:
                i_svd += 1
            if i_svd < len(svd_list):
                mid = svd_list[i_svd][0]
                seq.append(("svd", mid, len(seq)+1))
                used.add(mid)
                need_svd -= 1
                i_svd += 1
            turn = "p5"
        else:
            # If the current turn model has no movies left, switch to the opposite turn model
            turn = "p5" if turn == "svd" else "svd"
            # If neither can move, end.
            if (need_p5 <= 0 or i_p5 >= len(p5_list)) and (need_svd <= 0 or i_svd >= len(svd_list)):
                break

    return seq  # max length = 10

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
    return rows

def upsert_rows(rows: List[Dict]):
    if rows:
        sb.table("recommendations").upsert(rows).execute()

# ===================== API =====================
class RecReq(BaseModel):
    session_id: str
    topk_per_model: int = 10
    phase: int = 2

@app.on_event("startup")
def _startup():
    load_svd()
    load_tokenizer_once()
    load_base_state_once()
    load_datamaps()
    print("[startup] SVD and P5 tokenizer/state cached.")

@app.get("/health")
def health():
    return {
        "ok": True,
        "device": DEVICE,
        "p5_ckpt": P5_CKPT,
        "svd_dir": SVD_DIR,
        "n_items": int(len(ITEM_IDS) if ITEM_IDS is not None else 0)
    }

@app.post("/recommend")
def recommend(req: RecReq, x_webhook_secret: Optional[str] = Header(None)):
    if WEBHOOK_SECRET and x_webhook_secret != WEBHOOK_SECRET:
        raise HTTPException(401, "bad secret")

    # 1) Input acquisition
    hist = get_history(req.session_id)
    if not hist:
        raise HTTPException(400, "No rating history for the given session_id.")
    all_candidates = get_candidates()
    if not all_candidates:
        raise HTTPException(400, "No candidates found for phase2_movies.")

    # 2) SVD Top-100 Scores
    p_u = infer_user_vec_svd(hist)
    svd_scored_all = score_candidates_svd(p_u, all_candidates) if p_u is not None else []
    svd_top100 = sorted(svd_scored_all, key=lambda x: x[1], reverse=True)[:100]
    svd_rows = rows_from_scored(req.session_id, "svd", svd_top100, topk=req.topk_per_model, phase=req.phase)

    # 3) P5 soft prompt: create per-user base + attach adapter + short finetuning
    hist_mapped = map_history_for_p5(hist, max_n=HISTORY_MAX_TRAIN) # mapping according to datamaps
    base = create_per_request_base()
    per_user = attach_soft_prompt(base, TOKENIZER)
    finetune_soft_prompt(per_user, TOKENIZER, req.session_id, hist_mapped)

    # 4) P5: Rerank only SVD Top-100 (Using trained per_user model)
    pairs = []  # [(ext_mid, internal_id)]
    for ext_mid, _ in svd_top100:
        internal = map_movie_for_p5(ext_mid)
        if internal is not None:
            pairs.append((ext_mid, internal))
    
    if pairs:
        mapped_ids = [internal for _, internal in pairs]
        p5_scores = p5_score_candidates_mapped(per_user, TOKENIZER, req.session_id, hist_mapped, mapped_ids)
        # Pair score with external movie_id
        p5_scored_on_100 = [(ext_mid, float(sc)) for (ext_mid, _), sc in zip(pairs, p5_scores)]
    else:
        p5_scored_on_100 = []
        
    p5_rows = rows_from_scored(req.session_id, "p5", p5_scored_on_100, topk=req.topk_per_model, phase=req.phase)

    # 5) Display order
    p5_top_pairs = [(r["movie_id"], r["score"]) for r in sorted(p5_rows, key=lambda x:x["rank"])][:5]
    svd_top_pairs = [(r["movie_id"], r["score"]) for r in sorted(svd_rows, key=lambda x:x["rank"])][:5]
    display_seq = build_display_sequence(p5_top_pairs, svd_top_pairs)

    # 6) Reflect display order
    disp_map = {(m, mid): order for (m, mid, order) in display_seq}
    for r in p5_rows:
        key = ("p5", r["movie_id"])
        if key in disp_map:
            r["display_order"] = disp_map[key]
    for r in svd_rows:
        key = ("svd", r["movie_id"])
        if key in disp_map:
            r["display_order"] = disp_map[key]

    # 7) upsert
    upsert_rows(p5_rows)
    upsert_rows(svd_rows)

    return {
        "session_id": req.session_id,
        "phase": req.phase,
        "svd_top_saved": len(svd_rows),    # 10
        "p5_top_saved": len(p5_rows),      # 10
        "svd_top100_size": len(svd_top100),
        "display_sequence": display_seq
    }
