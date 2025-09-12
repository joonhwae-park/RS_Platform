/*
  # Add attitude questions for serendipity, novelty, and diversity

  1. New Columns
    - `serendipity_attitude` (text) - Consumer attitude toward serendipitous recommendations (1-7 scale)
    - `novelty_attitude` (text) - Consumer attitude toward novel recommendations (1-7 scale)  
    - `diversity_attitude` (text) - Consumer attitude toward diverse recommendations (1-7 scale)

  2. Changes
    - Add three new required columns to questionnaire_responses table
    - All columns are text type to store the scale values (1-7)
*/

-- Add new columns for attitude toward recommendation characteristics
DO $$
BEGIN
  -- Add serendipity_attitude column if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'serendipity_attitude'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN serendipity_attitude text;
  END IF;

  -- Add novelty_attitude column if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'novelty_attitude'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN novelty_attitude text;
  END IF;

  -- Add diversity_attitude column if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'diversity_attitude'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN diversity_attitude text;
  END IF;
END $$;