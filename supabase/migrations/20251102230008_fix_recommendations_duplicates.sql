/*
  # Fix recommendations table duplicates

  1. Changes
    - Remove duplicate recommendations keeping only the latest ones
    - Add unique constraint to prevent future duplicates on (session_id, movie_id, model, phase)
    - Create index on (session_id, created_at) for faster queries

  2. Important Notes
    - This migration cleans up duplicate data before adding constraints
    - The unique constraint ensures one recommendation per movie/model/phase per session
*/

-- Step 1: Delete older duplicate recommendations, keeping only the most recent ones
DELETE FROM recommendations
WHERE id IN (
  SELECT id FROM (
    SELECT 
      id,
      ROW_NUMBER() OVER (
        PARTITION BY session_id, movie_id, model, phase 
        ORDER BY created_at DESC
      ) as rn
    FROM recommendations
  ) t
  WHERE t.rn > 1
);

-- Step 2: Add unique constraint to prevent future duplicates
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_constraint 
    WHERE conname = 'recommendations_unique_session_movie_model_phase'
  ) THEN
    ALTER TABLE recommendations 
    ADD CONSTRAINT recommendations_unique_session_movie_model_phase 
    UNIQUE (session_id, movie_id, model, phase);
  END IF;
END $$;

-- Step 3: Create index for faster queries by session and created_at
CREATE INDEX IF NOT EXISTS idx_recommendations_session_created 
ON recommendations(session_id, created_at DESC);

-- Step 4: Create index for faster queries by session and display_order
CREATE INDEX IF NOT EXISTS idx_recommendations_session_display_order 
ON recommendations(session_id, display_order) 
WHERE display_order IS NOT NULL;
