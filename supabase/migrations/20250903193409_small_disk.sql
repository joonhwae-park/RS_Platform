/*
  # Add email field for lottery participation

  1. New Column
    - `email` (text, nullable) - Optional email address for lottery reward delivery

  2. Changes
    - Add email column to questionnaire_responses table
    - Column is optional to maintain compatibility with existing data
*/

-- Add email column for lottery participation
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'questionnaire_responses' AND column_name = 'email'
  ) THEN
    ALTER TABLE questionnaire_responses ADD COLUMN email text;
  END IF;
END $$;