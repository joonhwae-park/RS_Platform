/*
  # Add attention check column to questionnaire responses

  1. Changes
    - Add `attention_check` column to `questionnaire_responses` table
    - Column is required (NOT NULL) to ensure attention check is completed
*/

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'attention_check'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN attention_check text NOT NULL DEFAULT '';
  END IF;
END $$;