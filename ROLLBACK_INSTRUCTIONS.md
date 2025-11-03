# Rollback Instructions: Soft Prompt → In-Prompt Approach

## Current State
- **Soft Prompt ACTIVE**: P5 model fine-tunes soft prompts on user history
- **Backup Location**: `app.py.in-prompt-backup`

## To Rollback to In-Prompt Approach

If soft prompt is not performing well, restore the in-prompt version:

```bash
# Option 1: Quick restore (overwrites current app.py)
cp app.py.in-prompt-backup app.py

# Option 2: Safe restore (keeps current version as backup)
mv app.py app.py.soft-prompt-backup
cp app.py.in-prompt-backup app.py
```

## Key Differences

### In-Prompt Approach (Backed Up)
- **Method**: Uses base P5 model without finetuning
- **Personalization**: User history included in prompt text
- **Speed**: Faster (no training step)
- **Lines 617-624**: Creates base model → Immediately uses for scoring

### Soft Prompt Approach (Current)
- **Method**: Fine-tunes soft prompt embeddings on user history
- **Personalization**: Model adapts to user preferences via gradient updates
- **Speed**: Slower (~10 training steps per request)
- **Lines 617-643**: Creates base model → Attaches PEFT adapter → Fine-tunes → Uses for scoring
- **Added**:
  - Lines 617-619: Loads `mvt_aug_epoch10.pth` pretrained weights
  - Lines 621-624: Attaches soft prompt adapter via PEFT
  - Lines 626-629: Calls `finetune_soft_prompt()` on adapter
  - Line 643: Uses `per_user_model` (not base) for scoring

## Testing Both Approaches

To compare effectiveness:
1. **Deploy Soft Prompt version** (current)
2. Monitor P5 ranking quality and diversity
3. If underwhelming: Restore in-prompt version
4. Compare: Do rankings differ significantly from SVD?

## Files Involved

- `app.py` - Current version (soft prompt active)
- `app.py.in-prompt-backup` - In-prompt version (no finetuning)
- `app.py.soft-prompt-backup` - Created if you rollback (optional)
