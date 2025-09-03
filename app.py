# app.py
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
P5_MAX_LEN = int(os.getenv("P5_MAX_LEN", "512"))          # Aligned with test_ml1m_small.py
P5_GEN_MAX_LEN = int(os.getenv("P5_GEN_MAX_LEN", "64"))   # generate(max_length=...)
P5_DROPOUT = float(os.getenv("P5_DROPOUT", "0.1"))
P5_BATCH = int(os.getenv("P5_BATCH", "16"))

SVD_DIR = os.getenv("SVD_DIR", "/models/svd/v1")  # Directory where SVD files are stored (e.g., V.npy, ...)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    MU = json.load(open(f"{SVD_DIR}/global_mean.json")).get("mu", 0.0) if os.path.exists(f"{SVD_DIR}/global_mean.json") else 0.0
    REG = json.load(open(f"{SVD_DIR}/lambda.json")).get("reg", 0.05) if os.path.exists(f"{SVD_DIR}/lambda.json") else 0.05
    IDX = {mid: i for i, mid in enumerate(ITEM_IDS)}

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

TOKENIZER, P5MODEL = None, None

def create_config_eval():
    cfg = T5Config.from_pretrained(P5_BACKBONE)
    cfg.dropout_rate = P5_DROPOUT
    cfg.dropout = P5_DROPOUT
    cfg.attention_dropout = P5_DROPOUT
    cfg.activation_dropout = P5_DROPOUT
    cfg.losses = "rating"
    return cfg

def load_p5():
    global TOKENIZER, P5MODEL
    TOKENIZER = P5Tokenizer.from_pretrained(P5_BACKBONE, max_length=P5_MAX_LEN, do_lower_case=False)
    cfg = create_config_eval()
    P5MODEL = P5Pretraining.from_pretrained(P5_BACKBONE, config=cfg).to(DEVICE)
    P5MODEL.resize_token_embeddings(TOKENIZER.vocab_size)
    P5MODEL.tokenizer = TOKENIZER
    P5MODEL.eval()
    if os.path.exists(P5_CKPT):
        state = load_state_dict(P5_CKPT, DEVICE)
        _ = P5MODEL.load_state_dict(state, strict=False)
        print(f"[P5] checkpoint loaded from {P5_CKPT}")
    else:
        print(f"[P5] WARNING: checkpoint not found at {P5_CKPT}")

def _first_float(text: str, default: float = -1.0) -> float:
    m = re.search(r"-?\d+(\.\d+)?", text.strip())
    return float(m.group(0)) if m else default

def make_p5_prompt(session_id: str, movie_id: str, history: List[Dict]) -> str:
    hist_str = ", ".join([f"movie_{h['movie_id']}:{float(h['ratings']):.1f}" for h in history[:10]]) or "none"
    return (
        f"Task: rating prediction.\n"
        f"user_session_{session_id} history: {hist_str}\n"
        f"Question: what rating would user_session_{session_id} give to movie_{movie_id} ?\n"
        f"Answer:"
    )

@torch.no_grad()
def p5_score_candidates(session_id: str, history: List[Dict], cand_ids: List[str]) -> List[Tuple[str, float]]:
    res: List[Tuple[str, float]] = []
    texts = [make_p5_prompt(session_id, cid, history) for cid in cand_ids]
    for s in range(0, len(texts), P5_BATCH):
        batch = texts[s:s+P5_BATCH]
        enc = TOKENIZER(batch, return_tensors="pt", padding=True, truncation=True, max_length=P5_MAX_LEN)
        enc = {k: v.to(DEVICE) for k,v in enc.items()}
        out = P5MODEL.generate(**enc, max_length=P5_GEN_MAX_LEN, num_beams=1)
        dec = TOKENIZER.batch_decode(out, skip_special_tokens=True)
        for cid, txt in zip(cand_ids[s:s+P5_BATCH], dec):
            res.append((cid, _first_float(txt, default=-1.0)))
    return res

# ===================== Supabase I/O =====================
def get_history(session_id: str, limit: int = 50) -> List[Dict]:
    r = sb.table("movie_ratings")\
        .select("movie_id,ratings")\
        .eq("session_id", session_id)\
        .order("created_at", desc=True)\
        .limit(limit).execute()
    return r.data or []

def get_candidates(limit: int = 2000) -> List[str]:
    r = sb.table("phase2_movies").select("id").limit(limit).execute()
    return [row["id"] for row in (r.data or [])]

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
    load_p5()
    print("[startup] SVD and P5 loaded.")

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
    cand_ids = get_candidates()
    if not cand_ids:
        raise HTTPException(400, "No candidates found for phase2_movies.")

    # 2) SVD Top-10 Scores
    p_u = infer_user_vec_svd(hist)
    svd_scored = score_candidates_svd(p_u, cand_ids) if p_u is not None else []
    svd_rows = rows_from_scored(req.session_id, "svd", svd_scored, req.topk_per_model, req.phase)

    # 3) P5 Top-10 Scores
    p5_scored = p5_score_candidates(req.session_id, hist, cand_ids)
    p5_rows = rows_from_scored(req.session_id, "p5", p5_scored, req.topk_per_model, req.phase)

    # 4) Create a 10-item display order (= top 5 from P5 + top 5 from SVD) - Skip duplicates, interleave the two lists, and use a random start.
    #    Extract the top list for each model and pass it on
    p5_top_pairs = [(r["movie_id"], r["score"]) for r in sorted(p5_rows, key=lambda x:x["rank"])][:5]
    svd_top_pairs = [(r["movie_id"], r["score"]) for r in sorted(svd_rows, key=lambda x:x["rank"])][:5]
    display_seq = build_display_sequence(p5_top_pairs, svd_top_pairs)
    # display_seq: [(model, movie_id, display_order), ...]

    # 5) Reflect display_order
    #    Assign display_order to each model’s 10 rows; keep others as None.
    disp_map = {(m, mid): order for (m, mid, order) in display_seq}
    for r in p5_rows:
        key = ("p5", r["movie_id"])
        if key in disp_map:
            r["display_order"] = disp_map[key]
    for r in svd_rows:
        key = ("svd", r["movie_id"])
        if key in disp_map:
            r["display_order"] = disp_map[key]

    # Upsert: 20 rows (10 per model); display 10 with order 1..10, rest NULL.
    upsert_rows(p5_rows)
    upsert_rows(svd_rows)

    return {
        "session_id": req.session_id,
        "phase": req.phase,
        "svd_top_saved": len(svd_rows),
        "p5_top_saved": len(p5_rows),
        "display_sequence": display_seq
    }
