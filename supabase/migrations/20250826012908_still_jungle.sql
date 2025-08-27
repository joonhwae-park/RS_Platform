/*
  # Add columns for new questionnaire questions

  1. New Columns
    - `openness_to_experience` (text) - stores responses for openness to experience question (1-7 scale)
    - `risk_aversion` (text) - stores responses for risk aversion question (1-7 scale)
    - `movie_expertise` (text) - stores responses for movie expertise question (1-7 scale)
    - `attention_check` (text) - stores responses for attention check question (1-7 scale)

  2. Changes
    - All columns are nullable to maintain compatibility with existing data
    - Uses text type to store scale responses as strings
</*/

-- Add new columns for psychological measures and attention check
DO $$
BEGIN
  -- Add openness_to_experience column if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'openness_to_experience'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN openness_to_experience text;
  END IF;

  -- Add risk_aversion column if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'risk_aversion'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN risk_aversion text;
  END IF;

  -- Add movie_expertise column if it doesn't exist
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'movie_expertise'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN movie_expertise text;
  END IF;

  -- Update attention_check column to ensure it exists and has proper default
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'attention_check'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN attention_check text DEFAULT '';
  ELSE
    -- Update existing column to remove default if needed
    ALTER TABLE questionnaire_responses ALTER COLUMN attention_check DROP DEFAULT;
    ALTER TABLE questionnaire_responses ALTER COLUMN attention_check SET NOT NULL;
  END IF;
END $$;